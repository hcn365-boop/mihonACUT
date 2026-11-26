package eu.kanade.tachiyomi.ui.reader.colorize

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.os.SystemClock
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.roundToInt
import logcat.LogPriority
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import tachiyomi.core.common.util.system.logcat

/**
 * Runs the Manga Light V6 generator (v6_generator.onnx) on a page.
 *
 * Dynamically scales input dimensions to preserve aspect ratio within [MIN_SIDE..MAX_SIDE]
 * (aligned to multiples of [STRIDE]) when the ONNX model supports dynamic spatial shapes,
 * falling back to 512x512 if the model requires fixed spatial dimensions.
 *
 * Not thread safe: call it from one coroutine at a time.
 */
class MangaColorizer(
    modelFile: File,
    acceleration: Acceleration = Acceleration.CPU,
    threads: Int = 0,
    private val guide: SamFeatures? = null,
) : Closeable {
    /**
     * Predicted RGB at inference resolution ([width] x [height]),
     * interleaved (R,G,B per pixel), values in [0, 1].
     */
    class Chroma(
        val width: Int,
        val height: Int,
        val rgb: FloatArray,
    )

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val supportsDynamicSize: Boolean

    private val zeroSam0: FloatArray? = if (guide?.level0 == null) FloatArray(256 * 64 * 64) else null
    private val zeroSam1: FloatArray? = if (guide?.level1 == null) FloatArray(256 * 32 * 32) else null
    private val zeroWd14 = FloatArray(1024)

    // Reusable 1x3 Rec.601 luminance kernel.
    private val lumaWeights = floatArrayOf(0.299f, 0.587f, 0.114f)

    init {
        check(OpenCVLoader.initLocal()) { "OpenCV native library failed to load" }
        session = OrtSessions.create(env, modelFile.absolutePath, acceleration, threads)
        val expectedInputs = setOf(INPUT_IMAGE, INPUT_SAM0, INPUT_SAM1, INPUT_WD14)
        if (session.inputNames != expectedInputs || session.outputNames != setOf(OUTPUT_RGB)) {
            val message = "Not a Manga Light V6 generator: inputs=${session.inputNames}, outputs=${session.outputNames}"
            session.close()
            error(message)
        }
        val imageInfo = session.inputInfo[INPUT_IMAGE]?.info as? TensorInfo
        val shape = imageInfo?.shape
        // Check if H and W dimensions in [N, C, H, W] are dynamic (<= 0)
        supportsDynamicSize = shape != null && shape.size == 4 && shape[2] <= 0L && shape[3] <= 0L
        logcat(LogPriority.DEBUG) {
            "[Perf] MangaColorizer initialized: inputShape=${shape?.joinToString()}, dynamicSize=$supportsDynamicSize"
        }
    }

    private inline fun <T> measureStep(name: String, block: () -> T): T {
        val startMs = SystemClock.elapsedRealtime()
        return try {
            block()
        } finally {
            val elapsedMs = SystemClock.elapsedRealtime() - startMs
            logcat(LogPriority.DEBUG) { "[Perf] MangaColorizer.$name took ${elapsedMs}ms" }
        }
    }

    fun predict(input: Bitmap): Chroma = measureStep("predict(${input.width}x${input.height})") {
        Mats().use { m ->
            val rgba = m.mat()
            toMat(input, rgba)
            predictFromRgba(m, rgba)
        }
    }

    internal fun predictFromRgba(m: Mats, rgba: Mat): Chroma =
        measureStep("predictFromRgba(${rgba.cols()}x${rgba.rows()})") {
            val (inferW, inferH) = computeInferenceSize(rgba.cols(), rgba.rows(), supportsDynamicSize)
            val plane = inferW * inferH

            val gray = m.mat()
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            val small = m.mat()
            PageResizer.resizeMatLanczos2(gray, small, inferW, inferH)
            m.free(gray)

            // Model input range is [-1, 1].
            small.convertTo(small, CvType.CV_32F, 1.0 / 127.5, -1.0)
            val modelInputBuffer = FloatArray(plane)
            small.get(0, 0, modelInputBuffer)
            m.free(small)

            val modelOutputBuffer = FloatArray(3 * plane)
            measureStep("onnx.runModel(${inferW}x${inferH})") {
                runModel(modelInputBuffer, inferW, inferH, modelOutputBuffer)
            }

            val planes = ArrayList<Mat>(3)
            for (c in 0 until 3) {
                val planeMat = m.mat(inferH, inferW, CvType.CV_32F)
                val start = c * plane
                planeMat.put(0, 0, modelOutputBuffer.copyOfRange(start, start + plane))
                planes.add(planeMat)
            }
            val merged = m.mat(inferH, inferW, CvType.CV_32FC3)
            Core.merge(planes, merged)
            planes.forEach { m.free(it) }

            // Model output range is [-1, 1]; bring it to [0, 1] and clamp.
            merged.convertTo(merged, CvType.CV_32FC3, 0.5, 0.5)
            Core.max(merged, ZERO_SCALAR_3, merged)
            Core.min(merged, ONE_SCALAR_3, merged)

            val chromaOutBuffer = FloatArray(3 * plane)
            merged.get(0, 0, chromaOutBuffer)
            Chroma(inferW, inferH, chromaOutBuffer)
        }

    /**
     * Builds the colored page while reducing low-resolution upscaled lighting blur on text and sharp lines.
     * Alpha is preserved.
     */
    fun render(input: Bitmap, chroma: Chroma, intensity: Float = 1f): Bitmap {
        require(intensity in 0f..1.2f) { "intensity must be within 0..1.2" }
        return measureStep("render(${input.width}x${input.height}, chroma=${chroma.width}x${chroma.height}, intensity=$intensity)") {
            Mats().use { m ->
                val rgba = m.mat()
                toMat(input, rgba)
                renderFromRgba(m, rgba, input.width, input.height, chroma, intensity)
            }
        }
    }

    internal fun renderFromRgba(
        m: Mats,
        rgba: Mat,
        width: Int,
        height: Int,
        chroma: Chroma,
        intensity: Float = 1f,
    ): Bitmap {
        require(intensity in 0f..1.2f) { "intensity must be within 0..1.2" }
        require(chroma.width > 0 && chroma.height > 0 && chroma.rgb.size == chroma.width * chroma.height * 3) {
            "Invalid chroma dimensions ${chroma.width}x${chroma.height} for buffer size ${chroma.rgb.size}"
        }
        return measureStep("renderFromRgba(${width}x${height}, chroma=${chroma.width}x${chroma.height}, intensity=$intensity)") {
            val fullSize = Size(width.toDouble(), height.toDouble())

            // Original luminance (Rec.601), full size, computed directly from RGBA without extra RGB8 copy.
            val rgb = m.mat()
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            rgb.convertTo(rgb, CvType.CV_32FC3, 1.0 / 255.0)

            val guideY = m.mat()
            luma(m, rgb, guideY)

            // Model color and luminance at dynamic model resolution (rows = height, cols = width).
            val modelRgb = m.mat(chroma.height, chroma.width, CvType.CV_32FC3)
            modelRgb.put(0, 0, chroma.rgb)
            val modelY = m.mat()
            luma(m, modelRgb, modelY)

            // Upscale model color and luminance directly to page size.
            val fullRgb = m.mat()
            Imgproc.resize(modelRgb, fullRgb, fullSize, 0.0, 0.0, Imgproc.INTER_LINEAR)
            m.free(modelRgb)

            val fullY = m.mat()
            Imgproc.resize(modelY, fullY, fullSize, 0.0, 0.0, Imgproc.INTER_LINEAR)
            m.free(modelY)

            // Saturation: scale chroma around upscaled model luminance, before normalizing.
            val colorY3 = m.mat()
            Core.merge(listOf(fullY, fullY, fullY), colorY3)

            val satColor = m.mat()
            Core.subtract(fullRgb, colorY3, satColor)
            m.free(fullRgb)

            val intensityScalar = Scalar(intensity.toDouble(), intensity.toDouble(), intensity.toDouble())
            Core.multiply(satColor, intensityScalar, satColor)
            Core.add(satColor, colorY3, satColor)
            Core.max(satColor, ZERO_SCALAR_3, satColor)
            Core.min(satColor, ONE_SCALAR_3, satColor)

            // Normalize to luminance 1 so only the hue/saturation remains.
            val satY = m.mat()
            luma(m, satColor, satY)
            Core.max(satY, Y_EPS_SCALAR_1, satY)

            val satY3 = m.mat()
            Core.merge(listOf(satY, satY, satY), satY3)
            m.free(satY)

            val hue = rgb // Reuse `rgb` Mat for normalized hue.
            Core.divide(satColor, satY3, hue)
            m.free(satColor)
            m.free(satY3)

            // 2. Model lighting (upscaled from model inference resolution).
            val light = fullY // Reuse `fullY` Mat for model lighting.
            Core.max(light, LIGHT_FLOOR_SCALAR_1, light)

            // 3. Detail ratio from original luminance vs local blur.
            val lowGuide = m.mat()
            Imgproc.blur(guideY, lowGuide, BLUR_KERNEL_SIZE)
            Core.max(lowGuide, LIGHT_FLOOR_SCALAR_1, lowGuide)

            val detail = m.mat()
            Core.divide(guideY, lowGuide, detail)
            m.free(lowGuide)

            Core.min(detail, ONE_SCALAR_1, detail)
            Core.max(detail, ZERO_SCALAR_1, detail)

            val detailShaped = m.mat()
            Core.pow(detail, DETAIL_STRENGTH.toDouble(), detailShaped)
            m.free(detail)

            // 4. Weight model lighting by original luminance guideY so dark text/strokes
            //    are not blurred by low-res upscaled `light`.
            val darkMask = m.mat()
            Core.min(guideY, ONE_SCALAR_1, darkMask)
            Core.max(darkMask, ZERO_SCALAR_1, darkMask)

            // lightWithDetail = light * detailShaped (reusing `light` buffer)
            Core.multiply(light, detailShaped, light)

            // sharpGuideWithDetail = guideY * detailShaped (reusing `detailShaped` buffer)
            Core.multiply(guideY, detailShaped, detailShaped)
            m.free(guideY)

            // invMask = 1.0 - darkMask (reusing `colorY3` buffer)
            val invMask = colorY3
            Core.multiply(darkMask, NEG_ONE_SCALAR_1, invMask)
            Core.add(invMask, ONE_SCALAR_1, invMask)

            // partLight = light * darkMask (reusing `light` buffer)
            Core.multiply(light, darkMask, light)
            m.free(darkMask)

            // partSharp = sharpGuideWithDetail * invMask (reusing `detailShaped` buffer)
            Core.multiply(detailShaped, invMask, detailShaped)
            m.free(invMask)

            // finalY = partLight + partSharp (reusing `light` buffer)
            val finalY = light
            Core.add(light, detailShaped, finalY)
            m.free(detailShaped)

            val finalY3 = m.mat()
            Core.merge(listOf(finalY, finalY, finalY), finalY3)
            m.free(finalY)

            val result = m.mat()
            Core.multiply(hue, finalY3, result)
            m.free(hue)
            m.free(finalY3)
            Core.max(result, ZERO_SCALAR_3, result)
            Core.min(result, ONE_SCALAR_3, result)

            val out8 = m.mat()
            result.convertTo(out8, CvType.CV_8UC3, 255.0)
            m.free(result)

            val out = m.mat()
            Imgproc.cvtColor(out8, out, Imgproc.COLOR_RGB2RGBA)
            m.free(out8)

            val alpha = m.mat()
            Core.extractChannel(rgba, alpha, 3)
            Core.insertChannel(alpha, out, 3)
            m.free(alpha)

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(out, bitmap)
            bitmap
        }
    }

    override fun close() {
        session.close()
    }

    /** Runs the ONNX model and writes 3 consecutive [inferW]x[inferH] planes (R, G, B) in [-1, 1] into [outValues]. */
    private fun runModel(image: FloatArray, inferW: Int, inferH: Int, outValues: FloatArray) {
        val plane = inferW * inferH
        val tensors = ArrayList<OnnxTensor>(4)
        try {
            fun tensor(data: FloatArray, vararg shape: Long): OnnxTensor =
                OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).also { tensors.add(it) }

            val inputs = mapOf(
                INPUT_IMAGE to tensor(image, 1, 1, inferH.toLong(), inferW.toLong()),
                INPUT_SAM0 to tensor(guide?.level0 ?: zeroSam0!!, 1, 256, 64, 64),
                INPUT_SAM1 to tensor(guide?.level1 ?: zeroSam1!!, 1, 256, 32, 32),
                INPUT_WD14 to tensor(zeroWd14, 1, 1024),
            )
            session.run(inputs).use { result ->
                val output = result[0] as OnnxTensor
                val expected = longArrayOf(1, 3, inferH.toLong(), inferW.toLong())
                if (!output.info.shape.contentEquals(expected)) {
                    logcat(LogPriority.ERROR) {
                        "MangaColorizer: unexpected output shape (got ${output.info.shape.joinToString()}, expected ${expected.joinToString()})"
                    }
                }
                require(output.info.shape.contentEquals(expected)) {
                    "Unexpected model output shape: ${output.info.shape.joinToString()}"
                }
                val buf = output.floatBuffer
                buf.rewind()
                buf.get(outValues, 0, 3 * plane)
            }
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e) { "MangaColorizer: failed to run model at ${inferW}x${inferH}" }
            throw e
        } finally {
            tensors.forEach { it.close() }
        }
    }

    /** Rec.601 luminance: 0.299R + 0.587G + 0.114B. Input CV_32FC3, output CV_32FC1. */
    private fun luma(m: Mats, rgb: Mat, dst: Mat) {
        val kernel = m.mat(1, 3, CvType.CV_32F)
        kernel.put(0, 0, lumaWeights)
        Core.transform(rgb, dst, kernel)
        m.free(kernel)
    }

    internal fun toMat(bitmap: Bitmap, destination: Mat) {
        val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) {
            bitmap
        } else {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
        try {
            Utils.bitmapToMat(argb, destination)
        } finally {
            if (argb !== bitmap) argb.recycle()
        }
    }

    /** Releases every Mat created through it, even if processing throws. */
    internal class Mats : Closeable {
        private val owned = ArrayList<Mat>(32)

        fun mat(): Mat = Mat().also { owned.add(it) }

        fun mat(rows: Int, cols: Int, type: Int): Mat = Mat(rows, cols, type).also { owned.add(it) }

        fun free(mat: Mat) {
            if (owned.remove(mat)) {
                mat.release()
            }
        }

        override fun close() {
            for (i in owned.indices) {
                owned[i].release()
            }
            owned.clear()
        }
    }

    companion object {
        const val MIN_SIDE = 512
        const val MAX_SIDE = 768
        const val STRIDE = 64

        /**
         * Computes aspect-ratio-preserving dimensions within [MIN_SIDE..MAX_SIDE],
         * snapped to multiples of [STRIDE] (64) to match U-Net downsampling strides.
         * Falls back to 512x512 if the ONNX graph has static spatial dimensions.
         */
        fun computeInferenceSize(srcWidth: Int, srcHeight: Int, dynamic: Boolean = true): Pair<Int, Int> {
            if (!dynamic || srcWidth <= 0 || srcHeight <= 0) {
                return MIN_SIDE to MIN_SIDE
            }
            val aspect = srcWidth.toDouble() / srcHeight.toDouble()
            var targetW: Double
            var targetH: Double

            if (aspect >= 1.0) {
                // Landscape or square: anchor shorter side (height) to MIN_SIDE, cap longer side (width) at MAX_SIDE
                targetH = MIN_SIDE.toDouble()
                targetW = targetH * aspect
                if (targetW > MAX_SIDE) {
                    targetW = MAX_SIDE.toDouble()
                    targetH = targetW / aspect
                }
            } else {
                // Portrait: anchor shorter side (width) to MIN_SIDE, cap longer side (height) at MAX_SIDE
                targetW = MIN_SIDE.toDouble()
                targetH = targetW / aspect
                if (targetH > MAX_SIDE) {
                    targetH = MAX_SIDE.toDouble()
                    targetW = targetH * aspect
                }
            }

            val snappedW = ((targetW / STRIDE).roundToInt() * STRIDE).coerceIn(MIN_SIDE, MAX_SIDE)
            val snappedH = ((targetH / STRIDE).roundToInt() * STRIDE).coerceIn(MIN_SIDE, MAX_SIDE)
            return snappedW to snappedH
        }

        private const val INPUT_IMAGE = "L_bw"
        private const val INPUT_SAM0 = "sam_level0"
        private const val INPUT_SAM1 = "sam_level1"
        private const val INPUT_WD14 = "wd14_embedding"
        private const val OUTPUT_RGB = "rgb_pred"

        private const val LIGHT_RADIUS = 16
        private const val LIGHT_FLOOR = 0.1
        private const val Y_EPS = 1e-3

        val BLUR_KERNEL_SIZE = Size(2.0 * LIGHT_RADIUS + 1.0, 2.0 * LIGHT_RADIUS + 1.0)
        val ZERO_SCALAR_1 = Scalar(0.0)
        val ONE_SCALAR_1 = Scalar(1.0)
        val NEG_ONE_SCALAR_1 = Scalar(-1.0)
        val LIGHT_FLOOR_SCALAR_1 = Scalar(LIGHT_FLOOR)
        val Y_EPS_SCALAR_1 = Scalar(Y_EPS)
        val ZERO_SCALAR_3 = Scalar(0.0, 0.0, 0.0)
        val ONE_SCALAR_3 = Scalar(1.0, 1.0, 1.0)

        /** Below 1.0 softens the original page's detail so it cannot overpower the printed color. */
        const val DETAIL_STRENGTH = 0.99
    }
}
