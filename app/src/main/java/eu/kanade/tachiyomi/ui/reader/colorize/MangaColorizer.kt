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
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import tachiyomi.core.common.util.system.logcat
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer

/**
 * Runs the Manga Light V6 generator on a page.
 *
 * The model predicts color at 512x512. During rendering, its color information is
 * combined with the original page's full-resolution luminance so text and ink
 * details are not replaced by the model's lower-resolution output.
 */
class MangaColorizer(
    modelFile: File,
    acceleration: Acceleration = Acceleration.CPU,
    threads: Int = 0,
    private val guide: SamFeatures? = null,
) : Closeable {

    /** Predicted RGB planes at 512x512, with values in [0, 255]. */
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
            val message =
                "Not a Manga Light V6 generator: inputs=${session.inputNames}, outputs=${session.outputNames}"
            session.close()
            error(message)
        }
    }

    fun predict(input: Bitmap): Chroma = Mats().use { mats ->
        val rgba = mats.mat()
        toMat(input, rgba)

        val gray = mats.mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)

        val small = mats.mat()
        Imgproc.resize(
            gray,
            small,
            Size(SIDE.toDouble(), SIDE.toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_AREA,
        )

        // Match the model's expected input range: [-1, 1].
        small.convertTo(small, CvType.CV_32F, 1.0 / 127.5, -1.0)

        val pixels = FloatArray(PLANE)
        small.get(0, 0, pixels)

        val rgbPlanes = runModel(pixels)

        // Convert model output from [-1, 1] to [0, 255].
        val rPlane = FloatArray(PLANE) { index ->
            ((rgbPlanes[index] + 1f) * 127.5f).coerceIn(0f, 255f)
        }
        val gPlane = FloatArray(PLANE) { index ->
            ((rgbPlanes[PLANE + index] + 1f) * 127.5f).coerceIn(0f, 255f)
        }
        val bPlane = FloatArray(PLANE) { index ->
            ((rgbPlanes[2 * PLANE + index] + 1f) * 127.5f).coerceIn(0f, 255f)
        }

        Chroma(r = rPlane, g = gPlane, b = bPlane)
    }

    /**
     * Applies model color while retaining the original page's full-resolution
     * luminance. This keeps text and fine ink lines sharp.
     *
     * [intensity] 0 keeps the original page; 1 uses the model's full color.
     */
    fun render(input: Bitmap, chroma: Chroma, intensity: Float = 1f): Bitmap {
        require(intensity in 0f..1.2f) { "intensity must be within 0..1.2" }
        require(chroma.r.size == PLANE && chroma.g.size == PLANE && chroma.b.size == PLANE) {
            "Chroma planes must each contain $PLANE values"
        }

        val width = input.width
        val height = input.height

        return Mats().use { mats ->
            val rgba = mats.mat()
            toMat(input, rgba)

            val originalRgb = mats.mat()
            Imgproc.cvtColor(rgba, originalRgb, Imgproc.COLOR_RGBA2RGB)

            // Intensity zero is an exact copy of the original RGB pixels.
            val resultRgb = mats.mat()
            if (intensity <= 0f) {
                originalRgb.copyTo(resultRgb)
            } else {
                val rSmall = mats.mat(SIDE, SIDE, CvType.CV_32F).apply {
                    put(0, 0, chroma.r)
                }
                val gSmall = mats.mat(SIDE, SIDE, CvType.CV_32F).apply {
                    put(0, 0, chroma.g)
                }
                val bSmall = mats.mat(SIDE, SIDE, CvType.CV_32F).apply {
                    put(0, 0, chroma.b)
                }

                val predictedSmall = mats.mat()
                Core.merge(listOf(rSmall, gSmall, bSmall), predictedSmall)

                val predictedSmall8 = mats.mat()
                predictedSmall.convertTo(predictedSmall8, CvType.CV_8UC3)

                val predictedRgb = mats.mat()
                Imgproc.resize(
                    predictedSmall8,
                    predictedRgb,
                    Size(width.toDouble(), height.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_LANCZOS4,
                )

                // Lab separates luminance (L) from color (a, b).
                val originalLab = mats.mat()
                Imgproc.cvtColor(originalRgb, originalLab, Imgproc.COLOR_RGB2Lab)

                val predictedLab = mats.mat()
                Imgproc.cvtColor(predictedRgb, predictedLab, Imgproc.COLOR_RGB2Lab)

                val originalChannels = ArrayList<Mat>(3)
                val predictedChannels = ArrayList<Mat>(3)
                Core.split(originalLab, originalChannels)
                Core.split(predictedLab, predictedChannels)

                val colorA = mats.mat()
                val colorB = mats.mat()

                val modelWeight = intensity.toDouble()
                val originalWeight = 1.0 - modelWeight

                // Blend only color channels. Original L is kept at full resolution.
                Core.addWeighted(
                    predictedChannels[1],
                    modelWeight,
                    originalChannels[1],
                    originalWeight,
                    0.0,
                    colorA,
                )
                Core.addWeighted(
                    predictedChannels[2],
                    modelWeight,
                    originalChannels[2],
                    originalWeight,
                    0.0,
                    colorB,
                )

                val outputLab = mats.mat()
                Core.merge(
                    listOf(originalChannels[0], colorA, colorB),
                    outputLab,
                )
                Imgproc.cvtColor(outputLab, resultRgb, Imgproc.COLOR_Lab2RGB)

                originalChannels.forEach { it.release() }
                predictedChannels.forEach { it.release() }
            }

            val outputRgba = mats.mat()
            Imgproc.cvtColor(resultRgb, outputRgba, Imgproc.COLOR_RGB2RGBA)

            // Retain the input alpha channel, if present.
            val alpha = mats.mat()
            Core.extractChannel(rgba, alpha, 3)
            Core.insertChannel(alpha, outputRgba, 3)

            Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
                Utils.matToBitmap(outputRgba, it)
            }
        }
    }

    override fun close() {
        session.close()
    }

    /** Returns three consecutive 512x512 planes (R, G, B), with values in [-1, 1]. */
    private fun runModel(image: FloatArray): FloatArray {
        val tensors = ArrayList<OnnxTensor>()

        try {
            fun tensor(data: FloatArray, vararg shape: Long): OnnxTensor =
                OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).also(tensors::add)

            val inputs = mapOf(
                INPUT_IMAGE to tensor(image, 1, 1, SIDE.toLong(), SIDE.toLong()),
                INPUT_SAM0 to tensor(
                    guide?.level0 ?: FloatArray(256 * 64 * 64),
                    1,
                    256,
                    64,
                    64,
                ),
                INPUT_SAM1 to tensor(
                    guide?.level1 ?: FloatArray(256 * 32 * 32),
                    1,
                    256,
                    32,
                    32,
                ),
                INPUT_WD14 to tensor(FloatArray(1024), 1, 1024),
            )

            session.run(inputs).use { result ->
                val output = result[0] as OnnxTensor
                val expectedShape = longArrayOf(1, 3, SIDE.toLong(), SIDE.toLong())

                if (!output.info.shape.contentEquals(expectedShape)) {
                    logcat(LogPriority.ERROR) {
                        "Generator: unexpected output shape " +
                            "(${output.info.shape.joinToString()})"
                    }
                }

                require(output.info.shape.contentEquals(expectedShape)) {
                    "Unexpected model output shape: ${output.info.shape.joinToString()}"
                }

                return FloatArray(3 * PLANE).also {
                    output.floatBuffer.get(it)
                }
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
            if (argb !== bitmap) {
                argb.recycle()
            }
        }
    }

    private class Mats : Closeable {
        private val owned = ArrayList<Mat>()

        fun mat(): Mat = Mat().also(owned::add)

        fun mat(rows: Int, cols: Int, type: Int): Mat =
            Mat(rows, cols, type).also(owned::add)

        override fun close() {
            owned.forEach(Mat::release)
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
