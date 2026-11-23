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
 *  - [render] uses model hue/lighting for broad areas while preserving sharp original luminance
 *    around fine lines and text to avoid low-res upscaled lighting blur.
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
        val gray = m.mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        val small = m.mat()
        Imgproc.resize(gray, small, Size(SIDE.toDouble(), SIDE.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        // Model input range is [-1, 1].
        small.convertTo(small, CvType.CV_32F, 1.0 / 127.5, -1.0)
        val pixels = FloatArray(PLANE)
        small.get(0, 0, pixels)

        val rgbPlanes = runModel(pixels)
        val planes = List(3) { c ->
            m.mat(SIDE, SIDE, CvType.CV_32F).also { it.put(0, 0, rgbPlanes.copyOfRange(c * PLANE, (c + 1) * PLANE)) }
        }
        val merged = m.mat(SIDE, SIDE, CvType.CV_32FC3)
        Core.merge(planes, merged)
        // Model output range is [-1, 1]; bring it to [0, 1] and clamp.
        merged.convertTo(merged, CvType.CV_32FC3, 0.5, 0.5)
        Core.max(merged, Scalar(0.0, 0.0, 0.0), merged)
        Core.min(merged, Scalar(1.0, 1.0, 1.0), merged)

        val out = FloatArray(3 * PLANE)
        merged.get(0, 0, out)
        Chroma(out)
    }

    /**
     * Builds the colored page while avoiding low-resolution upscaled lighting blur on text and sharp lines.
     *  1. Hue: upscaled model color normalized to luminance 1 after saturation.
     *  2. Light: upscaled model luminance with floor applied.
     *  3. Detail: local contrast ratio of original luminance relative to its blur.
     *  4. Final luminance blends model lighting and sharp original luminance based on [detail],
     *     preventing blurry upscaled lighting from creating halos around dark text and lines.
     * Alpha is preserved.
     */
    fun render(input: Bitmap, chroma: Chroma, intensity: Float = 1f): Bitmap {
        require(intensity in 0f..1.2f) { "intensity must be within 0..1.2" }
        val width = input.width
        val height = input.height
        val fullSize = Size(width.toDouble(), height.toDouble())
        return Mats().use { m ->
            val rgba = m.mat()
            toMat(input, rgba)
            val rgb8 = m.mat()
            Imgproc.cvtColor(rgba, rgb8, Imgproc.COLOR_RGBA2RGB)
            val rgb = m.mat()
            rgb8.convertTo(rgb, CvType.CV_32FC3, 1.0 / 255.0)

            // Original luminance (Rec.601), full size.
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
            val fullY = m.mat()
            Imgproc.resize(modelY, fullY, fullSize, 0.0, 0.0, Imgproc.INTER_LINEAR)

            // Saturation: scale chroma around each pixel's gray value, before normalizing.
            val colorY = m.mat()
            luma(m, fullRgb, colorY)
            val colorY3 = m.mat()
            Core.merge(listOf(colorY, colorY, colorY), colorY3)
            val satColor = m.mat()
            Core.subtract(fullRgb, colorY3, satColor)
            Core.multiply(satColor, Scalar(intensity.toDouble(), intensity.toDouble(), intensity.toDouble()), satColor)
            Core.add(satColor, colorY3, satColor)
            Core.max(satColor, Scalar(0.0, 0.0, 0.0), satColor)
            Core.min(satColor, Scalar(1.0, 1.0, 1.0), satColor)

            // Normalize to luminance 1 so only the hue/saturation remains.
            val satY = m.mat()
            luma(m, satColor, satY)
            Core.max(satY, Scalar(Y_EPS), satY)
            val satY3 = m.mat()
            Core.merge(listOf(satY, satY, satY), satY3)
            val hue = m.mat()
            Core.divide(satColor, satY3, hue)

            // 2. Model lighting (upscaled from 512x512).
            val light = m.mat()
            fullY.copyTo(light)
            Core.max(light, Scalar(LIGHT_FLOOR), light)

            // 3. Detail ratio from original luminance vs local blur.
            val lowGuide = m.mat()
            Imgproc.blur(guideY, lowGuide, Size(2.0 * LIGHT_RADIUS + 1.0, 2.0 * LIGHT_RADIUS + 1.0))
            Core.max(lowGuide, Scalar(LIGHT_FLOOR), lowGuide)
            val detail = m.mat()
            Core.divide(guideY, lowGuide, detail)

            val ones = m.mat(detail.rows(), detail.cols(), detail.type())
            ones.setTo(Scalar(1.0))
            Core.min(detail, ones, detail)
            Core.max(detail, Scalar(0.0), detail)

            val detailShaped = m.mat()
            Core.pow(detail, DETAIL_STRENGTH.toDouble(), detailShaped)

            // 4. Blend: where detailShaped < 1 (dark text/lines), preserve sharp original guideY;
            //    where detailShaped == 1, use model light * detailShaped.
            //    finalY = detailShaped * (light * detailShaped) + (1 - detailShaped) * guideY
            val modelCombined = m.mat()
            Core.multiply(light, detailShaped, modelCombined)

            val fromModel = m.mat()
            Core.multiply(modelCombined, detailShaped, fromModel)

            val invDetail = m.mat()
            Core.subtract(ones, detailShaped, invDetail)
            val fromGuide = m.mat()
            Core.multiply(guideY, invDetail, fromGuide)

            val finalY = m.mat()
            Core.add(fromModel, fromGuide, finalY)

            val finalY3 = m.mat()
            Core.merge(listOf(finalY, finalY, finalY), finalY3)

            val result = m.mat()
            Core.multiply(hue, finalY3, result)
            Core.max(result, Scalar(0.0, 0.0, 0.0), result)
            Core.min(result, Scalar(1.0, 1.0, 1.0), result)

            val out8 = m.mat()
            result.convertTo(out8, CvType.CV_8UC3, 255.0)
            val out = m.mat()
            Imgproc.cvtColor(out8, out, Imgproc.COLOR_RGB2RGBA)
            val alpha = m.mat()
            Core.extractChannel(rgba, alpha, 3)
            Core.insertChannel(alpha, out, 3)

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(out, bitmap)
            bitmap
        }
    }

    override fun close() {
        session.close()
    }

    /** Returns the model output as 3 consecutive 512x512 planes (R, G, B), values in [-1, 1]. */
    private fun runModel(image: FloatArray): FloatArray {
        val tensors = ArrayList<OnnxTensor>()
        try {
            fun tensor(data: FloatArray, vararg shape: Long): OnnxTensor =
                OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).also { tensors.add(it) }

            val inputs = mapOf(
                INPUT_IMAGE to tensor(image, 1, 1, SIDE.toLong(), SIDE.toLong()),
                INPUT_SAM0 to tensor(guide?.level0 ?: FloatArray(256 * 64 * 64), 1, 256, 64, 64),
                INPUT_SAM1 to tensor(guide?.level1 ?: FloatArray(256 * 32 * 32), 1, 256, 32, 32),
                INPUT_WD14 to tensor(FloatArray(1024), 1, 1024),
            )
            session.run(inputs).use { result ->
                val output = result[0] as OnnxTensor
                val expected = longArrayOf(1, 3, SIDE.toLong(), SIDE.toLong())
                if (!output.info.shape.contentEquals(expected)) {
                    logcat(LogPriority.ERROR) { "SAM encoder: unexpected shapes or failed to output expected shape (got ${output.info.shape.joinToString()})" }
                }
                require(output.info.shape.contentEquals(expected)) {
                    "Unexpected model output shape: ${output.info.shape.joinToString()}"
                }
                val values = FloatArray(3 * PLANE)
                output.floatBuffer.get(values)
                return values
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
        kernel.put(0, 0, floatArrayOf(0.299f, 0.587f, 0.114f))
        Core.transform(rgb, dst, kernel)
    }

    private fun toMat(bitmap: Bitmap, destination: Mat) {
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
    private class Mats : Closeable {
        private val owned = ArrayList<Mat>()

        fun mat(): Mat = Mat().also { owned.add(it) }

        fun mat(rows: Int, cols: Int, type: Int): Mat = Mat(rows, cols, type).also { owned.add(it) }

        override fun close() {
            owned.forEach { it.release() }
        }
    }

    private companion object {
        const val SIDE = 512
        const val PLANE = SIDE * SIDE
        const val INPUT_IMAGE = "L_bw"
        const val INPUT_SAM0 = "sam_level0"
        const val INPUT_SAM1 = "sam_level1"
        const val INPUT_WD14 = "wd14_embedding"
        const val OUTPUT_RGB = "rgb_pred"

        const val LIGHT_RADIUS = 16
        const val LIGHT_FLOOR = 0.1
        const val Y_EPS = 1e-3

        /** Below 1.0 softens the original page's detail so it cannot overpower the printed color. */
        const val DETAIL_STRENGTH = 0.99
    }
}
