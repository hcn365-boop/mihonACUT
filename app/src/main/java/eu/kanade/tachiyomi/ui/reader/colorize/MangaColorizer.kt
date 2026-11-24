package eu.kanade.tachiyomi.ui.reader.colorize

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
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
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer

/**
 * Runs the Manga Light V6 generator (v6_generator.onnx) on a page.
 *
 * Two steps so the expensive part can be cached:
 *  - [predict] runs the model once and returns the predicted RGB at model resolution (small).
 *  - [render] uses model hue/lighting while weighting model lighting by original luminance
 *    so dark text and fine strokes do not inherit low-resolution upscaled lighting blur.
 *
 * Not thread safe: call it from one coroutine at a time.
 */
class MangaColorizer(
    modelFile: File,
    acceleration: Acceleration = Acceleration.CPU,
    threads: Int = 0,
    private val guide: SamFeatures? = null,
) : Closeable {
    /** Predicted RGB at 512x512, interleaved (R,G,B per pixel), values in [0, 1]. About 3 MB per page. */
    class Chroma(val rgb: FloatArray)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private val zeroSam0: FloatArray? = if (guide?.level0 == null) FloatArray(256 * 64 * 64) else null
    private val zeroSam1: FloatArray? = if (guide?.level1 == null) FloatArray(256 * 32 * 32) else null
    private val zeroWd14 = FloatArray(1024)

    // Reusable buffers for 512x512 inference to avoid repeated allocations across pages.
    private val modelInputBuffer = FloatArray(PLANE)
    private val modelOutputBuffer = FloatArray(3 * PLANE)
    private val chromaOutBuffer = FloatArray(3 * PLANE)

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
    }

    fun predict(input: Bitmap): Chroma = Mats().use { m ->
        val rgba = m.mat()
        toMat(input, rgba)
        predictFromRgba(m, rgba)
    }

    internal fun predictFromRgba(m: Mats, rgba: Mat): Chroma {
        val gray = m.mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        val small = m.mat()
        Imgproc.resize(gray, small, SIDE_SIZE, 0.0, 0.0, Imgproc.INTER_AREA)
        m.free(gray)

        // Model input range is [-1, 1].
        small.convertTo(small, CvType.CV_32F, 1.0 / 127.5, -1.0)
        small.get(0, 0, modelInputBuffer)
        m.free(small)

        runModel(modelInputBuffer, modelOutputBuffer)

        val planes = ArrayList<Mat>(3)
        for (c in 0 until 3) {
            val planeMat = m.mat(SIDE, SIDE, CvType.CV_32F)
            planeMat.put(0, 0, modelOutputBuffer, c * PLANE, PLANE)
            planes.add(planeMat)
        }
        val merged = m.mat(SIDE, SIDE, CvType.CV_32FC3)
        Core.merge(planes, merged)
        planes.forEach { m.free(it) }

        // Model output range is [-1, 1]; bring it to [0, 1] and clamp.
        merged.convertTo(merged, CvType.CV_32FC3, 0.5, 0.5)
        Core.max(merged, ZERO_SCALAR_3, merged)
        Core.min(merged, ONE_SCALAR_3, merged)

        merged.get(0, 0, chromaOutBuffer)
        return Chroma(chromaOutBuffer.copyOf())
    }

    /**
     * Builds the colored page while reducing low-resolution upscaled lighting blur on text and sharp lines.
     *  1. Hue: upscaled model color normalized to luminance 1 after saturation.
     *  2. Light: upscaled model luminance with floor applied.
     *  3. Detail: local contrast ratio of original luminance relative to its blur.
     *  4. Final luminance weights model lighting by original luminance [guideY] so dark strokes/text
     *     retain crisp original luminance while brighter/mid areas use model lighting + detail.
     * Alpha is preserved.
     */
    fun render(input: Bitmap, chroma: Chroma, intensity: Float = 1f): Bitmap {
        require(intensity in 0f..1.2f) { "intensity must be within 0..1.2" }
        return Mats().use { m ->
            val rgba = m.mat()
            toMat(input, rgba)
            renderFromRgba(m, rgba, input.width, input.height, chroma, intensity)
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
        val fullSize = Size(width.toDouble(), height.toDouble())

        // Original luminance (Rec.601), full size, computed directly from RGBA without extra RGB8 copy.
        val rgb = m.mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        rgb.convertTo(rgb, CvType.CV_32FC3, 1.0 / 255.0)

        val guideY = m.mat()
        luma(m, rgb, guideY)

        // Model color and luminance at model resolution.
        val modelRgb = m.mat(SIDE, SIDE, CvType.CV_32FC3)
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

        // 2. Model lighting (upscaled from 512x512).
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
        return bitmap
    }

    override fun close() {
        session.close()
    }

    /** Runs the ONNX model and writes 3 consecutive 512x512 planes (R, G, B) in [-1, 1] into [outValues]. */
    private fun runModel(image: FloatArray, outValues: FloatArray) {
        val tensors = ArrayList<OnnxTensor>(4)
        try {
            fun tensor(data: FloatArray, vararg shape: Long): OnnxTensor =
                OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).also { tensors.add(it) }

            val inputs = mapOf(
                INPUT_IMAGE to tensor(image, 1, 1, SIDE.toLong(), SIDE.toLong()),
                INPUT_SAM0 to tensor(guide?.level0 ?: zeroSam0!!, 1, 256, 64, 64),
                INPUT_SAM1 to tensor(guide?.level1 ?: zeroSam1!!, 1, 256, 32, 32),
                INPUT_WD14 to tensor(zeroWd14, 1, 1024),
            )
            session.run(inputs).use { result ->
                val output = result[0] as OnnxTensor
                val expected = longArrayOf(1, 3, SIDE.toLong(), SIDE.toLong())
                if (!output.info.shape.contentEquals(expected)) {
                    logcat(LogPriority.ERROR) {
                        "SAM encoder: unexpected shapes or failed to output expected shape (got ${output.info.shape.joinToString()})"
                    }
                }
                require(output.info.shape.contentEquals(expected)) {
                    "Unexpected model output shape: ${output.info.shape.joinToString()}"
                }
                val buf = output.floatBuffer
                buf.rewind()
                buf.get(outValues, 0, 3 * PLANE)
            }
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e) { "SAM encoder: failed to process features" }
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

    private companion object {
        const val SIDE = 512
        const val PLANE = SIDE * SIDE
        val SIDE_SIZE = Size(SIDE.toDouble(), SIDE.toDouble())

        const val INPUT_IMAGE = "L_bw"
        const val INPUT_SAM0 = "sam_level0"
        const val INPUT_SAM1 = "sam_level1"
        const val INPUT_WD14 = "wd14_embedding"
        const val OUTPUT_RGB = "rgb_pred"

        const val LIGHT_RADIUS = 16
        const val LIGHT_FLOOR = 0.1
        const val Y_EPS = 1e-3

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
