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
 * Manga Light V6 colorizer.
 *
 * The model predicts RGB at 512x512. Its result is resized to the input bitmap's
 * dimensions. At full intensity, the model's color channels are retained, while
 * dark text/ink luminance from the original image is restored without sharpening
 * or blurring, which avoids introducing edge halos.
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
     * intensity = 1: full model color, with dark original ink/text restored.
     *
     * Values above 1 are accepted for compatibility, but clamped to 1.
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
        val amount = intensity.coerceIn(0f, 1f)

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
            mats.release(inputRgba)

            val resultRgb = mats.mat()

            if (amount <= 0f) {
                originalRgb.copyTo(resultRgb)
            } else {
                // Rebuild the model's 512x512 RGB output.
                val rSmall = mats.mat(SIDE, SIDE, CvType.CV_32F)
                rSmall.put(0, 0, chroma.r)

                val gSmall = mats.mat(SIDE, SIDE, CvType.CV_32F)
                gSmall.put(0, 0, chroma.g)

                val bSmall = mats.mat(SIDE, SIDE, CvType.CV_32F)
                bSmall.put(0, 0, chroma.b)

                val modelSmallFloat = mats.mat()
                Core.merge(
                    listOf(rSmall, gSmall, bSmall),
                    modelSmallFloat,
                )

                mats.release(rSmall)
                mats.release(gSmall)
                mats.release(bSmall)

                val modelSmallRgb = mats.mat()
                modelSmallFloat.convertTo(
                    modelSmallRgb,
                    CvType.CV_8UC3,
                )
                mats.release(modelSmallFloat)

                // Resize only the model output to the input bitmap dimensions.
                val modelRgb = mats.mat()
                Imgproc.resize(
                    modelSmallRgb,
                    modelRgb,
                    Size(width.toDouble(), height.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_LANCZOS4,
                )
                mats.release(modelSmallRgb)

                val colorRgb: Mat
                if (amount >= 1f) {
                    colorRgb = modelRgb
                } else {
                    colorRgb = mats.mat()
                    Core.addWeighted(
                        modelRgb,
                        amount.toDouble(),
                        originalRgb,
                        (1f - amount).toDouble(),
                        0.0,
                        colorRgb,
                    )
                    mats.release(modelRgb)
                }

                // Convert the colorized page and original to Lab.
                val colorLab = mats.mat()
                Imgproc.cvtColor(
                    colorRgb,
                    colorLab,
                    Imgproc.COLOR_RGB2Lab,
                )
                mats.release(colorRgb)

                val originalLab = mats.mat()
                Imgproc.cvtColor(
                    originalRgb,
                    originalLab,
                    Imgproc.COLOR_RGB2Lab,
                )
                mats.release(originalRgb)

                // The model's a/b color channels stay unchanged.
                // Only its L channel is adjusted to recover dark source ink.
                val modelL = mats.mat()
                Core.extractChannel(colorLab, modelL, 0)

                val originalL = mats.mat()
                Core.extractChannel(originalLab, originalL, 0)
                mats.release(originalLab)

                val pixelCount = Math.multiplyExact(width, height)
                val modelLight = ByteArray(pixelCount)
                val originalLight = ByteArray(pixelCount)

                modelL.get(0, 0, modelLight)
                originalL.get(0, 0, originalLight)

                for (i in 0 until pixelCount) {
                    val modelValue = modelLight[i].toInt() and 0xFF
                    val sourceValue = originalLight[i].toInt() and 0xFF

                    /*
                     * A smooth per-pixel ink mask:
                     * - dark source pixels get strong ink restoration;
                     * - light source pixels are left untouched.
                     *
                     * No blur, edge sharpening, or high-pass filter is used.
                     */
                    val t = (
                        (INK_MASK_LIGHT_LIMIT - sourceValue).toFloat() /
                            (INK_MASK_LIGHT_LIMIT - INK_MASK_DARK_LIMIT)
                        ).coerceIn(0f, 1f)

                    val inkMask = t * t * (3f - 2f * t)
                    val weight = INK_RESTORE_STRENGTH * amount * inkMask

                    // Only move toward a darker original value; never brighten
                    // an edge pixel, which helps avoid bright halos.
                    val restoredValue = if (sourceValue < modelValue) {
                        modelValue - (modelValue - sourceValue) * weight
                    } else {
                        modelValue.toFloat()
                    }

                    modelLight[i] =
                        (restoredValue + 0.5f)
                            .toInt()
                            .coerceIn(0, 255)
                            .toByte()
                }

                modelL.put(0, 0, modelLight)

                // Replace only Lab's L channel; retain the model's a/b channels.
                Core.insertChannel(modelL, colorLab, 0)
                mats.release(modelL)

                Imgproc.cvtColor(
                    colorLab,
                    resultRgb,
                    Imgproc.COLOR_Lab2RGB,
                )
                mats.release(colorLab)
                mats.release(originalL)
            }

            val outputRgba = mats.mat()
            Imgproc.cvtColor(
                resultRgb,
                outputRgba,
                Imgproc.COLOR_RGB2RGBA,
            )
            mats.release(resultRgb)

            Core.insertChannel(alpha, outputRgba, 3)
            mats.release(alpha)

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

        fun release(mat: Mat) {
            val index = owned.indexOfFirst { it === mat }
            if (index >= 0) {
                owned.removeAt(index)
                mat.release()
            }
        }

        override fun close() {
            owned.forEach { it.release() }
            owned.clear()
        }
    }

    private companion object {
        const val SIDE = 512
        const val PLANE = SIDE * SIDE

        // Smoothly restore source ink in dark regions only.
        const val INK_MASK_DARK_LIMIT = 95
        const val INK_MASK_LIGHT_LIMIT = 220
        const val INK_RESTORE_STRENGTH = 0.90f

        const val INPUT_IMAGE = "L_bw"
        const val INPUT_SAM0 = "sam_level0"
        const val INPUT_SAM1 = "sam_level1"
        const val INPUT_WD14 = "wd14_embedding"
        const val OUTPUT_RGB = "rgb_pred"
    }
}
