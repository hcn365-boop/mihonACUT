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
 * يشغّل نموذج Manga Light V6 لتلوين الصفحة.
 *
 * النموذج يتنبأ عند 512×512. عند العرض، تُكبّر النتيجة إلى أبعاد
 * الصورة المدخلة، ثم تُضاف إليها تفاصيل الإضاءة الدقيقة من الأصل
 * للمساعدة في الحفاظ على حدة النصوص والخطوط.
 */
class MangaColorizer(
    modelFile: File,
    acceleration: Acceleration = Acceleration.CPU,
    threads: Int = 0,
    private val guide: SamFeatures? = null,
) : Closeable {

    /**
     * قنوات RGB المتنبأ بها من النموذج، كل قناة بحجم 512×512
     * وقيمها في النطاق [0, 255].
     */
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
     * يجهّز الصورة الرمادية بالحجم الذي يتطلبه النموذج،
     * ثم يعيد تنبؤ النموذج في قنوات RGB.
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

        // تحويل نطاق الصورة من [0, 255] إلى [-1, 1].
        small.convertTo(
            small,
            CvType.CV_32F,
            1.0 / 127.5,
            -1.0,
        )

        val pixels = FloatArray(PLANE)
        small.get(0, 0, pixels)

        val modelOutput = runModel(pixels)

        // تحويل خرج النموذج من [-1, 1] إلى [0, 255].
        val r = FloatArray(PLANE) { index ->
            ((modelOutput[index] + 1f) * 127.5f)
                .coerceIn(0f, 255f)
        }

        val g = FloatArray(PLANE) { index ->
            ((modelOutput[PLANE + index] + 1f) * 127.5f)
                .coerceIn(0f, 255f)
        }

        val b = FloatArray(PLANE) { index ->
            ((modelOutput[2 * PLANE + index] + 1f) * 127.5f)
                .coerceIn(0f, 255f)
        }

        Chroma(r = r, g = g, b = b)
    }

    /**
     * يعرض نتيجة التلوين بأبعاد الصورة المدخلة.
     *
     * intensity = 0: الصورة الأصلية.
     * intensity = 1: ألوان وإضاءة خرج النموذج، مع استعادة التفاصيل الدقيقة
     *                من الصورة الأصلية.
     *
     * تُقبل القيم حتى 1.2 للتوافق مع الإعدادات الحالية، لكن تُحصر فعليًا
     * عند 1.0 لتجنب تجاوز نطاق المزج.
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

            val originalRgb = mats.mat()
            Imgproc.cvtColor(
                inputRgba,
                originalRgb,
                Imgproc.COLOR_RGBA2RGB,
            )

            val resultRgb = mats.mat()

            if (amount <= 0f) {
                originalRgb.copyTo(resultRgb)
            } else {
                // إعادة بناء خرج الموديل من قنوات RGB.
                val rSmall = mats.mat(SIDE, SIDE, CvType.CV_32F).apply {
                    put(0, 0, chroma.r)
                }
                val gSmall = mats.mat(SIDE, SIDE, CvType.CV_32F).apply {
                    put(0, 0, chroma.g)
                }
                val bSmall = mats.mat(SIDE, SIDE, CvType.CV_32F).apply {
                    put(0, 0, chroma.b)
                }

                val modelSmallFloat = mats.mat()
                Core.merge(
                    listOf(rSmall, gSmall, bSmall),
                    modelSmallFloat,
                )

                val modelSmallRgb = mats.mat()
                modelSmallFloat.convertTo(
                    modelSmallRgb,
                    CvType.CV_8UC3,
                )

                // تكبير خرج النموذج إلى أبعاد الصورة الداخلة.
                val modelRgb = mats.mat()
                Imgproc.resize(
                    modelSmallRgb,
                    modelRgb,
                    Size(width.toDouble(), height.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_LANCZOS4,
                )

                // فصل الإضاءة عن اللون باستخدام Lab.
                val modelLab = mats.mat()
                Imgproc.cvtColor(
                    modelRgb,
                    modelLab,
                    Imgproc.COLOR_RGB2Lab,
                )

                val originalLab = mats.mat()
                Imgproc.cvtColor(
                    originalRgb,
                    originalLab,
                    Imgproc.COLOR_RGB2Lab,
                )

                val modelL = mats.mat()
                val modelA = mats.mat()
                val modelB = mats.mat()
                Core.extractChannel(modelLab, modelL, 0)
                Core.extractChannel(modelLab, modelA, 1)
                Core.extractChannel(modelLab, modelB, 2)

                val originalL = mats.mat()
                val originalA = mats.mat()
                val originalB = mats.mat()
                Core.extractChannel(originalLab, originalL, 0)
                Core.extractChannel(originalLab, originalA, 1)
                Core.extractChannel(originalLab, originalB, 2)

                // مزج الإضاءة واللون حسب الشدة.
                // عند intensity = 1، تبقى قنوات اللون وإضاءة الموديل.
                val baseL = mats.mat()
                val baseA = mats.mat()
                val baseB = mats.mat()

                if (amount >= 1f) {
                    modelL.copyTo(baseL)
                    modelA.copyTo(baseA)
                    modelB.copyTo(baseB)
                } else {
                    val modelWeight = amount.toDouble()
                    val originalWeight = (1f - amount).toDouble()

                    Core.addWeighted(
                        modelL,
                        modelWeight,
                        originalL,
                        originalWeight,
                        0.0,
                        baseL,
                    )
                    Core.addWeighted(
                        modelA,
                        modelWeight,
                        originalA,
                        originalWeight,
                        0.0,
                        baseA,
                    )
                    Core.addWeighted(
                        modelB,
                        modelWeight,
                        originalB,
                        originalWeight,
                        0.0,
                        baseB,
                    )
                }

                // استخراج تفاصيل الإضاءة الدقيقة من الصورة الأصلية:
                // التفاصيل = الإضاءة الأصلية - نسخة مطموسة قليلًا منها.
                // نستخدم Float حتى تبقى التفاصيل السالبة والموجبة محفوظة.
                val originalLFloat = mats.mat()
                originalL.convertTo(
                    originalLFloat,
                    CvType.CV_32F,
                )

                val blurredOriginalL = mats.mat()
                Imgproc.GaussianBlur(
                    originalLFloat,
                    blurredOriginalL,
                    Size(5.0, 5.0),
                    DETAIL_SIGMA,
                    DETAIL_SIGMA,
                    Core.BORDER_REFLECT_101,
                )

                val originalDetails = mats.mat()
                Core.subtract(
                    originalLFloat,
                    blurredOriginalL,
                    originalDetails,
                )

                // إضافة تفاصيل الأصل فوق إضاءة الموديل فقط؛
                // لا نستبدل إضاءة الموديل بإضاءة الأصل كاملةً.
                val baseLFloat = mats.mat()
                baseL.convertTo(
                    baseLFloat,
                    CvType.CV_32F,
                )

                val detailedLFloat = mats.mat()
                Core.addWeighted(
                    baseLFloat,
                    1.0,
                    originalDetails,
                    (DETAIL_STRENGTH * amount).toDouble(),
                    0.0,
                    detailedLFloat,
                )

                // التحويل إلى 8-bit يحصر القيم تلقائيًا ضمن [0, 255].
                val detailedL = mats.mat()
                detailedLFloat.convertTo(
                    detailedL,
                    CvType.CV_8UC1,
                )

                val outputLab = mats.mat()
                Core.merge(
                    listOf(detailedL, baseA, baseB),
                    outputLab,
                )

                Imgproc.cvtColor(
                    outputLab,
                    resultRgb,
                    Imgproc.COLOR_Lab2RGB,
                )
            }

            val outputRgba = mats.mat()
            Imgproc.cvtColor(
                resultRgb,
                outputRgba,
                Imgproc.COLOR_RGB2RGBA,
            )

            // الاحتفاظ بقناة ألفا الأصلية.
            val alpha = mats.mat()
            Core.extractChannel(inputRgba, alpha, 3)
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
     * يعيد خرج النموذج في ثلاثة مستويات متتابعة: R ثم G ثم B،
     * وبالشكل [1, 3, 512, 512].
     */
    private fun runModel(image: FloatArray): FloatArray {
        val tensors = ArrayList<OnnxTensor>()

        try {
            fun tensor(
                data: FloatArray,
                vararg shape: Long,
            ): OnnxTensor {
                val result = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(data),
                    shape,
                )
                tensors.add(result)
                return result
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

        override fun close() {
            owned.forEach { it.release() }
        }
    }

    private companion object {
        const val SIDE = 512
        const val PLANE = SIDE * SIDE

        // مقدار معتدل لتفاصيل الأصل؛ زيادته كثيرًا قد تُظهر هالات حول النصوص.
        const val DETAIL_STRENGTH = 0.65f
        const val DETAIL_SIGMA = 1.0

        const val INPUT_IMAGE = "L_bw"
        const val INPUT_SAM0 = "sam_level0"
        const val INPUT_SAM1 = "sam_level1"
        const val INPUT_WD14 = "wd14_embedding"
        const val OUTPUT_RGB = "rgb_pred"
    }
}
