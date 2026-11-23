package eu.kanade.tachiyomi.ui.reader.colorize

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat

/**
 * Runs the Manga Light V6 generator (v6_generator.onnx) on a page.
 *
 * Two steps so the expensive part can be cached:
 *  - [predict] runs the model once and returns the predicted RGB at model resolution (small).
 *  - [render] takes luminance from the ORIGINAL page at full size and applies the model's
 *    color as a luminance-independent ratio (channel / model luminance).
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
        session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
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
     * Builds the colored page.
     * Luminance comes from [input] at full size. Color comes from the model as
     * ratio = model_RGB / model_Y, refined by a guided filter on the original luminance,
     * then scaled by [intensity] around neutral (1.0) and multiplied by the original luminance.
     * Alpha is preserved.
     */
    fun render(input: Bitmap, chroma: Chroma, intensity: Float = 1f): Bitmap {
        require(intensity in 0f..1.2f) { "intensity must be within 0..1.2" }
        val width = input.width
        val height = input.height
        return Mats().use { m ->
            val rgba = m.mat()
            toMat(input, rgba)
            val rgb8 = m.mat()
            Imgproc.cvtColor(rgba, rgb8, Imgproc.COLOR_RGBA2RGB)
            val rgb = m.mat()
            rgb8.convertTo(rgb, CvType.CV_32FC3, 1.0 / 255.0)

            // Original luminance (Rec.601), full size, used as the guide.
            val guideY = m.mat()
            luma(m, rgb, guideY)

            // Model color and its luminance at model resolution.
            val modelRgb = m.mat(SIDE, SIDE, CvType.CV_32FC3)
            modelRgb.put(0, 0, chroma.rgb)
            val modelY = m.mat()
            luma(m, modelRgb, modelY)
            val modelY3 = m.mat()
            Core.merge(listOf(modelY, modelY, modelY), modelY3)
            Core.add(modelY3, Scalar(RATIO_EPS, RATIO_EPS, RATIO_EPS), modelY3)

            // ratio = model_channel / model_Y, clamped to [0, MAX_RATIO].
            val ratio = m.mat()
            Core.divide(modelRgb, modelY3, ratio)
            Core.max(ratio, Scalar(0.0, 0.0, 0.0), ratio)
            Core.min(ratio, Scalar(MAX_RATIO, MAX_RATIO, MAX_RATIO), ratio)

            // Upscale the ratio to page size, then align its edges with the original luminance.
            val fullRatio = m.mat()
            Imgproc.resize(ratio, fullRatio, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            val channels = List(3) { m.mat() }
            Core.split(fullRatio, channels)
            val smoothChannels = channels.map { guidedFilter(m, guideY, it, GUIDE_RADIUS, GUIDE_EPS) }
            val smooth = m.mat()
            Core.merge(smoothChannels, smooth)

            // Saturation: move ratios away from neutral (1.0) by chromaScale.
            val adjusted = m.mat()
            Core.subtract(smooth, Scalar(1.0, 1.0, 1.0), adjusted)
            Core.multiply(adjusted, Scalar(intensity.toDouble(), intensity.toDouble(), intensity.toDouble()), adjusted)
            Core.add(adjusted, Scalar(1.0, 1.0, 1.0), adjusted)

            // Final color = adjusted ratio * original luminance.
            val guide3 = m.mat()
            Core.merge(listOf(guideY, guideY, guideY), guide3)
            val result = m.mat()
            Core.multiply(adjusted, guide3, result)
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
                // تم تعديل الأبعاد هنا لتتوافق مع مخرجات النموذج الجديد (64x64 و 32x32)
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

    /**
     * Guided filter (He et al.) with a single-channel guide [guide] and input [p], both CV_32FC1.
     * Edges of [p] follow edges of [guide].
     */
    private fun guidedFilter(m: Mats, guide: Mat, p: Mat, radius: Int, eps: Double): Mat {
        val ksize = Size(2.0 * radius + 1.0, 2.0 * radius + 1.0)
        fun box(src: Mat): Mat = m.mat().also { Imgproc.blur(src, it, ksize) }

        val meanI = box(guide)
        val meanP = box(p)

        val ii = m.mat()
        Core.multiply(guide, guide, ii)
        val ip = m.mat()
        Core.multiply(guide, p, ip)
        val meanII = box(ii)
        val meanIP = box(ip)

        val tmp = m.mat()
        Core.multiply(meanI, meanI, tmp)
        val varI = m.mat()
        Core.subtract(meanII, tmp, varI)

        Core.multiply(meanI, meanP, tmp)
        val cov = m.mat()
        Core.subtract(meanIP, tmp, cov)

        val denom = m.mat()
        Core.add(varI, Scalar(eps), denom)
        val a = m.mat()
        Core.divide(cov, denom, a)

        Core.multiply(a, meanI, tmp)
        val b = m.mat()
        Core.subtract(meanP, tmp, b)

        val meanA = box(a)
        val meanB = box(b)
        Core.multiply(meanA, guide, tmp)
        val q = m.mat()
        Core.add(tmp, meanB, q)
        return q
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

        const val RATIO_EPS = 1e-3
        const val MAX_RATIO = 4.0
        const val GUIDE_RADIUS = 8
        const val GUIDE_EPS = 1e-3
    }
}
