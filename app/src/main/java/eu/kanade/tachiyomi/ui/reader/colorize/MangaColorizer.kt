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
 *  - [render] takes its lighting from the MODEL, and only fine line detail from the ORIGINAL page.
 *    Printed color comes from the model, so the original page's overall brightness
 *    does not wash it out.
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
     *  1. Hue: model color smoothed along the original edges, then normalized to luminance 1.
     *  2. Light: the model's own luminance, guided by the original edges. This sets brightness and shadows.
     *  3. Detail: the original luminance divided by its large-scale blur. This keeps only lines and texture.
     *  4. Final luminance = light * detail. Saturation is applied to the color before normalization.
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

            // Original luminance (Rec.601), full size. Used for edges and line detail only.
            val guideY = m.mat()
            luma(m, rgb, guideY)

            // Model color and luminance at model resolution.
            val modelRgb = m.mat(SIDE, SIDE, CvType.CV_32FC3)
            modelRgb.put(0, 0, chroma.rgb)
            val modelY = m.mat()
            luma(m, modelRgb, modelY)

            // Upscale model color, luminance, and a confidence weight to page size.
            val weight = m.mat()
            Core.max(modelY, Scalar(WEIGHT_FLOOR), weight)
            val fullWeight = m.mat()
            Imgproc.resize(weight, fullWeight, fullSize, 0.0, 0.0, Imgproc.INTER_LINEAR)
            val fullRgb = m.mat()
            Imgproc.resize(modelRgb, fullRgb, fullSize, 0.0, 0.0, Imgproc.INTER_LINEAR)
            val fullY = m.mat()
            Imgproc.resize(modelY, fullY, fullSize, 0.0, 0.0, Imgproc.INTER_LINEAR)

            // 1. Hue: color smoothed along original edges, weighted by model luminance.
            val denom = guidedFilter(m, guideY, fullWeight, GUIDE_RADIUS, GUIDE_EPS)
            val channels = List(3) { m.mat() }
            Core.split(fullRgb, channels)
            val smoothChannels = channels.map { ch ->
                val weighted = m.mat()
                Core.multiply(ch, fullWeight, weighted)
                val num = guidedFilter(m, guideY, weighted, GUIDE_RADIUS, GUIDE_EPS)
                val out = m.mat()
                Core.divide(num, denom, out)
                out
            }
            val color = m.mat()
            Core.merge(smoothChannels, color)

            // Saturation: scale chroma around each pixel's gray value, before normalizing.
            val colorY = m.mat()
            luma(m, color, colorY)
            val colorY3 = m.mat()
            Core.merge(listOf(colorY, colorY, colorY), colorY3)
            val satColor = m.mat()
            Core.subtract(color, colorY3, satColor)
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

            // 2. Light: the model's own luminance, aligned to original edges. Sets brightness and shadows.
            val light = guidedFilter(m, guideY, fullY, GUIDE_RADIUS, GUIDE_EPS)
            Core.max(light, Scalar(LIGHT_FLOOR), light)

            // 3. Detail: original luminance relative to its large-scale blur. Keeps lines and texture only.
            val lowGuide = m.mat()
            Imgproc.blur(guideY, lowGuide, Size(2.0 * LIGHT_RADIUS + 1.0, 2.0 * LIGHT_RADIUS + 1.0))
            Core.max(lowGuide, Scalar(LIGHT_FLOOR), lowGuide)
            val detail = m.mat()
            Core.divide(guideY, lowGuide, detail)
            Core.pow(detail, DETAIL_STRENGTH, detail)

            // 4. Final luminance = model lighting * original detail.
            val finalY = m.mat()
            Core.multiply(light, detail, finalY)
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

        const val GUIDE_RADIUS = 8
        const val GUIDE_EPS = 1e-3
        const val WEIGHT_FLOOR = 0.05
        const val LIGHT_RADIUS = 32
        const val LIGHT_FLOOR = 0.02
        const val Y_EPS = 1e-3

        /** Below 1.0 softens the original page's detail so it cannot overpower the printed color. */
        const val DETAIL_STRENGTH = 0.5
    }
}
