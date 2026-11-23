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
 * Manga Light V6 colorizer.
 *
 * The model predicts RGB at 512x512 and only its color (a/b) is used.
 * Luminance always comes from the original image at full resolution, so
 * text and ink keep their sharpness. The color channels are upsampled with
 * a guided filter whose guide is the original luminance, so color edges
 * follow the original edges without bleeding onto text.
 */
class MangaColorizer(
    modelFile: File,
    acceleration: Acceleration = Acceleration.CPU,
    threads: Int = 0,
    private val guide: SamFeatures? = null,
) : Closeable {

    /** Model-predicted RGB planes at 512x512, values in [0, 255]. */
    class Chroma(
        val r: FloatArray,
        val g: FloatArray,
        val b: FloatArray,
    )

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        check(OpenCVLoader.initLocal()) {
            "OpenCV native library failed to load"
        }

        val options = OrtSession.SessionOptions().apply {
            if (threads > 0) {
                setIntraOpNumThreads(threads)
            }
        }

        session = env.createSession(modelFile.absolutePath, options)

        val expectedInputs = setOf(
            INPUT_IMAGE,
            INPUT_SAM0,
            INPUT_SAM1,
            INPUT_WD14,
        )

        if (
            session.inputNames != expectedInputs ||
            session.outputNames != setOf(OUTPUT_RGB)
        ) {
            val message =
                "Not a Manga Light V6 generator: " +
                    "inputs=${session.inputNames}, outputs=${session.outputNames}"

            session.close()
            error(message)
        }
    }

    /**
     * Converts the input to grayscale, resizes it to the model input size,
     * runs inference, and returns the predicted RGB planes.
     */
    fun predict(input: Bitmap): Chroma = Mats().use { mats ->
        val rgba = mats.mat()
        toMat(input, rgba)

        val gray = mats.mat()
        Imgproc.cvtColor(
            rgba,
            gray,
            Imgproc.COLOR_RGBA2GRAY,
        )

        val small = mats.mat()
        Imgproc.resize(
            gray,
            small,
            Size(SIDE.toDouble(), SIDE.toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_AREA,
        )

        // Normalize [0, 255] to [-1, 1].
        small.convertTo(
            small,
            CvType.CV_32F,
            1.0 / 127.5,
            -1.0,
        )

        val pixels = FloatArray(PLANE)
        small.get(0, 0, pixels)

        val output = runModel(pixels)

        // Denormalize model output from [-1, 1] to [0, 255].
        val r = FloatArray(PLANE) { index ->
            ((output[index] + 1f) * 127.5f).coerceIn(0f, 255f)
        }
        val g = FloatArray(PLANE) { index ->
            ((output[PLANE + index] + 1f) * 127.5f).coerceIn(0f, 255f)
        }
        val b = FloatArray(PLANE) { index ->
            ((output[2 * PLANE + index] + 1f) * 127.5f).coerceIn(0f, 255f)
        }

        Chroma(r = r, g = g, b = b)
    }

    /**
     * Renders the predicted colors at the input bitmap's dimensions.
     *
     * intensity = 0: original image.
     * intensity = 1: full model color with the original luminance.
     * Values up to 1.2 increase chroma saturation slightly.
     */
    fun render(
        input: Bitmap,
        chroma: Chroma,
        intensity: Float = 1f,
    ): Bitmap {
        require(intensity.isFinite()) {
            "intensity must be a finite number"
        }
        require(intensity in 0f..1.2f) {
            "intensity must be within 0..1.2"
        }
        require(
            chroma.r.size == PLANE &&
                chroma.g.size == PLANE &&
                chroma.b.size == PLANE,
        ) {
            "Each RGB plane must contain $PLANE values"
        }

        val width = input.width
        val height = input.height
        val chromaScale = intensity.toDouble()

        return Mats().use { mats ->
            val inputRgba = mats.mat()
            toMat(input, inputRgba)

            // Keep the source alpha channel for the output.
            val alpha = mats.mat()
            Core.extractChannel(inputRgba, alpha, 3)

            val originalRgb = mats.mat()
            Imgproc.cvtColor(
                inputRgba,
                originalRgb,
                Imgproc.COLOR_RGBA2RGB,
            )

            val resultRgb = mats.mat()

            if (intensity <= 0f) {
                originalRgb.copyTo(resultRgb)
            } else {
                // Rebuild the model's 512x512 RGB output as 8-bit.
                val rSmall = mats.mat(SIDE, SIDE, CvType.CV_32F)
                rSmall.put(0, 0, chroma.r)
                val gSmall = mats.mat(SIDE, SIDE, CvType.CV_32F)
                gSmall.put(0, 0, chroma.g)
                val bSmall = mats.mat(SIDE, SIDE, CvType.CV_32F)
                bSmall.put(0, 0, chroma.b)

                val modelSmallFloat = mats.mat()
                Core.merge(listOf(rSmall, gSmall, bSmall), modelSmallFloat)

                val modelSmallRgb = mats.mat()
                modelSmallFloat.convertTo(modelSmallRgb, CvType.CV_8UC3)

                // Resize only the color prediction to the input dimensions.
                val modelRgb = mats.mat()
                Imgproc.resize(
                    modelSmallRgb,
                    modelRgb,
                    Size(width.toDouble(), height.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_LANCZOS4,
                )

                // Original luminance at full resolution (Lab, float).
                val originalF = mats.mat()
                originalRgb.convertTo(originalF, CvType.CV_32FC3, 1.0 / 255.0)
                Imgproc.cvtColor(originalF, originalF, Imgproc.COLOR_RGB2Lab)

                // Model color in Lab (float).
                val modelF = mats.mat()
                modelRgb.convertTo(modelF, CvType.CV_32FC3, 1.0 / 255.0)
                Imgproc.cvtColor(modelF, modelF, Imgproc.COLOR_RGB2Lab)

                val originalChannels = ArrayList<Mat>()
                Core.split(originalF, originalChannels)
                val modelChannels = ArrayList<Mat>()
                Core.split(modelF, modelChannels)
                originalChannels.forEach { mats.own(it) }
                modelChannels.forEach { mats.own(it) }

                val lOrig = originalChannels[0] // 0..100
                val modelA = modelChannels[1]   // -127..127
                val modelB = modelChannels[2]

                // Guide = original luminance normalized to [0, 1].
                val guideMat = mats.mat()
                Core.multiply(lOrig, Scalar(0.01), guideMat)

                val aUp = guidedFilter(mats, guideMat, modelA)
                val bUp = guidedFilter(mats, guideMat, modelB)

                Core.multiply(aUp, Scalar(chromaScale), aUp)
                Core.multiply(bUp, Scalar(chromaScale), bUp)
                Core.max(aUp, Scalar(-127.0), aUp)
                Core.min(aUp, Scalar(127.0), aUp)
                Core.max(bUp, Scalar(-127.0), bUp)
                Core.min(bUp, Scalar(127.0), bUp)

                // Original L with the upsampled model a/b.
                val outLab = mats.mat()
                Core.merge(listOf(lOrig, aUp, bUp), outLab)

                val outRgbF = mats.mat()
                Imgproc.cvtColor(outLab, outRgbF, Imgproc.COLOR_Lab2RGB)
                outRgbF.convertTo(resultRgb, CvType.CV_8UC3, 255.0)
            }

            val outputRgba = mats.mat()
            Imgproc.cvtColor(
                resultRgb,
                outputRgba,
                Imgproc.COLOR_RGB2RGBA,
            )

            Core.insertChannel(alpha, outputRgba, 3)

            Bitmap.createBitmap(
                width,
                height,
                Bitmap.Config.ARGB_8888,
            ).also { bitmap ->
                Utils.matToBitmap(outputRgba, bitmap)
            }
        }
    }

    override fun close() {
        session.close()
    }

    /**
     * Guided filter: upsamples p (color channel) while aligning its edges
     * with the guide I (original luminance).
     */
    private fun guidedFilter(mats: Mats, guideI: Mat, p: Mat): Mat {
        val ksize = Size(2.0 * GUIDE_RADIUS + 1, 2.0 * GUIDE_RADIUS + 1)

        fun box(src: Mat): Mat {
            val dst = mats.mat()
            Imgproc.boxFilter(src, dst, -1, ksize)
            return dst
        }

        val meanI = box(guideI)
        val meanP = box(p)

        val ip = mats.mat()
        Core.multiply(guideI, p, ip)
        val meanIp = box(ip)

        val meanIMeanP = mats.mat()
        Core.multiply(meanI, meanP, meanIMeanP)
        val cov = mats.mat()
        Core.subtract(meanIp, meanIMeanP, cov)

        val ii = mats.mat()
        Core.multiply(guideI, guideI, ii)
        val meanII = box(ii)
        val meanISq = mats.mat()
        Core.multiply(meanI, meanI, meanISq)
        val variance = mats.mat()
        Core.subtract(meanII, meanISq, variance)
        Core.add(variance, Scalar(GUIDE_EPS), variance)

        val a = mats.mat()
        Core.divide(cov, variance, a)

        val aMeanI = mats.mat()
        Core.multiply(a, meanI, aMeanI)
        val b = mats.mat()
        Core.subtract(meanP, aMeanI, b)

        val meanA = box(a)
        val meanB = box(b)
        val aTimesI = mats.mat()
        Core.multiply(meanA, guideI, aTimesI)
        val q = mats.mat()
        Core.add(aTimesI, meanB, q)

        return q
    }

    /** Returns model output planes in R, G, B order, each in [-1, 1]. */
    private fun runModel(image: FloatArray): FloatArray {
        val tensors = ArrayList<OnnxTensor>()

        try {
            fun tensor(
                data: FloatArray,
                vararg shape: Long,
            ): OnnxTensor {
                val created = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(data),
                    shape,
                )
                tensors.add(created)
                return created
            }

            val inputs = mapOf(
                INPUT_IMAGE to tensor(
                    image,
                    1,
                    1,
                    SIDE.toLong(),
                    SIDE.toLong(),
                ),
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
                INPUT_WD14 to tensor(
                    FloatArray(1024),
                    1,
                    1024,
                ),
            )

            session.run(inputs).use { result ->
                val output = result[0] as OnnxTensor
                val expectedShape = longArrayOf(
                    1,
                    3,
                    SIDE.toLong(),
                    SIDE.toLong(),
                )

                if (!output.info.shape.contentEquals(expectedShape)) {
                    logcat(LogPriority.ERROR) {
                        "Generator: unexpected output shape " +
                            "(${output.info.shape.joinToString()})"
                    }
                }

                require(output.info.shape.contentEquals(expectedShape)) {
                    "Unexpected model output shape: " +
                        output.info.shape.joinToString()
                }

                return FloatArray(3 * PLANE).also { values ->
                    output.floatBuffer.get(values)
                }
            }
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e) {
                "Generator: failed to run inference"
            }
            throw e
        } finally {
            tensors.forEach { it.close() }
        }
    }

    private fun toMat(
        bitmap: Bitmap,
        destination: Mat,
    ) {
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

        fun mat(): Mat =
            Mat().also(owned::add)

        fun mat(
            rows: Int,
            cols: Int,
            type: Int,
        ): Mat =
            Mat(rows, cols, type).also(owned::add)

        /** Registers a Mat created elsewhere so it is released on close. */
        fun own(mat: Mat) {
            owned.add(mat)
        }

        override fun close() {
            owned.forEach { it.release() }
            owned.clear()
        }
    }

    private companion object {
        const val SIDE = 512
        const val PLANE = SIDE * SIDE

        const val GUIDE_RADIUS = 8
        const val GUIDE_EPS = 1e-3

        const val INPUT_IMAGE = "L_bw"
        const val INPUT_SAM0 = "sam_level0"
        const val INPUT_SAM1 = "sam_level1"
        const val INPUT_WD14 = "wd14_embedding"
        const val OUTPUT_RGB = "rgb_pred"
    }
}
