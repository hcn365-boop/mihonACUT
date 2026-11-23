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
 * Runs the Manga Light V6 generator (v6_generator.onnx) on a page,
 * matching the original Python ONNX inference pipeline:
 *  - Grayscale input resized to 512x512 (INTER_AREA) and normalized to [-1, 1].
 *  - Model output rgb_pred in [-1, 1] denormalized directly to RGB [0, 255].
 *  - Resized back to target page dimensions using INTER_LANCZOS4.
 */
class MangaColorizer(
    modelFile: File,
    acceleration: Acceleration = Acceleration.CPU,
    threads: Int = 0,
    private val guide: SamFeatures? = null,
) : Closeable {

    /**
     * Stores predicted RGB planes (R, G, B) at 512x512 in [0f, 255f] float space,
     * matching the original source denormalization without lossy Byte truncation.
     */
    class Chroma(
        val r: FloatArray,
        val g: FloatArray,
        val b: FloatArray,
    )

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        check(OpenCVLoader.initLocal()) { "OpenCV native library failed to load" }
        val options = OrtSession.SessionOptions().apply {
            if (threads > 0) {
                setIntraOpNumThreads(threads)
            }
        }
        session = env.createSession(modelFile.absolutePath, options)
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

        // Normalize to [-1, 1] just like (L_bw / 127.5) - 1.0 in inference.py
        small.convertTo(small, CvType.CV_32F, 1.0 / 127.5, -1.0)
        val pixels = FloatArray(PLANE)
        small.get(0, 0, pixels)

        val rgbPlanes = runModel(pixels)

        // Denormalize [-1, 1] -> [0, 255] and clamp, matching denormalize_rgb()
        val rPlane = FloatArray(PLANE) { i ->
            ((rgbPlanes[i] + 1.0f) * 127.5f).coerceIn(0f, 255f)
        }
        val gPlane = FloatArray(PLANE) { i ->
            ((rgbPlanes[PLANE + i] + 1.0f) * 127.5f).coerceIn(0f, 255f)
        }
        val bPlane = FloatArray(PLANE) { i ->
            ((rgbPlanes[2 * PLANE + i] + 1.0f) * 127.5f).coerceIn(0f, 255f)
        }

        Chroma(r = rPlane, g = gPlane, b = bPlane)
    }

    /**
     * Builds the colored page matching the original pipeline:
     * resizes predicted 512x512 RGB to full page size using INTER_LANCZOS4,
     * blends with [input] according to [intensity] (0 = original, 1 = full model color),
     * and preserves alpha channel if present.
     */
    fun render(input: Bitmap, chroma: Chroma, intensity: Float = 1f): Bitmap {
        require(intensity in 0f..1.2f) { "intensity must be within 0..1.2" }
        val width = input.width
        val height = input.height

        return Mats().use { m ->
            val rSmall = m.mat(SIDE, SIDE, CvType.CV_32F).apply { put(0, 0, chroma.r) }
            val gSmall = m.mat(SIDE, SIDE, CvType.CV_32F).apply { put(0, 0, chroma.g) }
            val bSmall = m.mat(SIDE, SIDE, CvType.CV_32F).apply { put(0, 0, chroma.b) }

            val mergedSmall = m.mat()
            Core.merge(listOf(rSmall, gSmall, bSmall), mergedSmall)

            val small8 = m.mat()
            mergedSmall.convertTo(small8, CvType.CV_8UC3)

            // Resize back to original page resolution using Lanczos4 (same as inference.py)
            val colorizedRgb = m.mat()
            Imgproc.resize(small8, colorizedRgb, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LANCZOS4)

            val rgba = m.mat()
            toMat(input, rgba)
            val origRgb = m.mat()
            Imgproc.cvtColor(rgba, origRgb, Imgproc.COLOR_RGBA2RGB)

            val blendedRgb = m.mat()
            when {
                intensity <= 0f -> origRgb.copyTo(blendedRgb)
                intensity == 1f -> colorizedRgb.copyTo(blendedRgb)
                else -> {
                    // Blend smoothly around full colorized output without washing out lines
                    Core.addWeighted(
                        colorizedRgb,
                        intensity.coerceIn(0f, 1.2f).toDouble(),
                        origRgb,
                        (1.0 - intensity.coerceIn(0f, 1.2f).coerceAtMost(1.0)),
                        0.0,
                        blendedRgb,
                    )
                }
            }

            val out = m.mat()
            Imgproc.cvtColor(blendedRgb, out, Imgproc.COLOR_RGB2RGBA)

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
                    logcat(LogPriority.ERROR) {
                        "Generator: unexpected output shape (got ${output.info.shape.joinToString()})"
                    }
                }
                require(output.info.shape.contentEquals(expected)) {
                    "Unexpected model output shape: ${output.info.shape.joinToString()}"
                }
                val values = FloatArray(3 * PLANE)
                output.floatBuffer.get(values)
                return values
            }
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e) { "Generator: failed to run inference" }
            throw e
        } finally {
            tensors.forEach { it.close() }
        }
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
    }
}
