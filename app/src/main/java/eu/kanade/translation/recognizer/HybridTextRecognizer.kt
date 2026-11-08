package eu.kanade.translation.recognizer

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import android.util.Log
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class HybridTextRecognizer(
    private val context: Context,
    override val language: TextRecognizerLanguage
) : TextRecognizer {

    private val TAG = "HybridTextRecognizer"

    private val mlKitRecognizer = MlKitTextRecognizer(language)
    private val paddleRecognizer = PaddleTextRecognizer(context, language, forceHorizontal = true)

    data class HybridBox(
        val x: Int,
        val y: Int,
        val width: Int,
        val height: Int,
        val angle: Float
    )

    data class HybridResult(
        val text: String,
        val box: HybridBox
    )

    override fun recognize(image: InputImage): String {
        val results = recognizeWithBoxes(image)
        return results.joinToString("\n") { it.text }
    }

    fun recognizeWithBoxes(image: InputImage): List<HybridResult> {
        val bitmap = image.bitmapInternal ?: return emptyList()
        val results = mutableListOf<HybridResult>()

        try {
            val mlKitResult: Text = mlKitRecognizer.process(image)

            for (block in mlKitResult.textBlocks) {
                val box = block.boundingBox ?: continue
                val line = block.lines.firstOrNull()
                val angle = line?.angle ?: 0f

                // حفظ الإحداثيات الأصلية من ML Kit بدقة متناهية دون أي إزاحة
                val originalBox = HybridBox(
                    x = box.left,
                    y = box.top,
                    width = box.width(),
                    height = box.height(),
                    angle = angle
                )

                // التثبت من الحدود أثناء القص فقط لضمان عدم الخروج عن حواف الصورة
                val startX = max(0, originalBox.x)
                val startY = max(0, originalBox.y)
                val endX = min(bitmap.width, originalBox.x + originalBox.width)
                val endY = min(bitmap.height, originalBox.y + originalBox.height)

                val cropW = endX - startX
                val cropH = endY - startY
                if (cropW < 10 || cropH < 10) continue

                var croppedBitmap: Bitmap? = null
                var rotatedBitmap: Bitmap? = null
                try {
                    croppedBitmap = Bitmap.createBitmap(bitmap, startX, startY, cropW, cropH)

                    // تدوير الصورة المقصوصة للتعرف عليها فقط بواسطة Paddle
                    rotatedBitmap = if (abs(angle) > 1f) {
                        val matrix = Matrix().apply {
                            postRotate(-angle, cropW / 2f, cropH / 2f)
                        }
                        Bitmap.createBitmap(croppedBitmap, 0, 0, cropW, cropH, matrix, true)
                    } else {
                        croppedBitmap
                    }

                    val croppedInput = InputImage.fromBitmap(rotatedBitmap, 0)
                    val paddleText = paddleRecognizer.recognize(croppedInput)

                    if (paddleText.isNotBlank()) {
                        // إرجاع النتيجة مع ربطها بـ originalBox (إحداثيات ML Kit الأصلية الثابتة)
                        results.add(HybridResult(paddleText.trim(), originalBox))
                    } else {
                        // في حال لم ينطق Paddle بنص، نستخدم نص ML Kit الأصلي بنفس الإحداثيات الأصلية
                        if (block.text.isNotBlank()) {
                            results.add(HybridResult(block.text.trim(), originalBox))
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "خطأ أثناء معالجة القصة الهجينة للصندوق", e)
                    if (block.text.isNotBlank()) {
                        results.add(HybridResult(block.text.trim(), originalBox))
                    }
                } finally {
                    if (rotatedBitmap != croppedBitmap) rotatedBitmap?.recycle()
                    croppedBitmap?.recycle()
                }
            }

            return results
        } catch (e: Exception) {
            Log.e(TAG, "فشل في تشغيل النظام الهجين", e)
            return emptyList()
        }
    }

    override fun close() {
        try {
            mlKitRecognizer.close()
            paddleRecognizer.close()
        } catch (e: Exception) {
            Log.e(TAG, "خطأ أثناء إغلاق محركات HybridTextRecognizer", e)
        }
    }
}
