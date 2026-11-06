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

    // محرك ML Kit لاستخراج إحداثيات وأماكن الصناديق هندسياً فقط
    private val mlKitRecognizer = MlKitTextRecognizer(language)

    // محرك PaddleOCR للتعرف الفعلي على النص حصرياً داخل الصناديق المقصوصة والمدمجة
    private val paddleRecognizer = PaddleTextRecognizer(context, language, forceHorizontal = true)

    // هيكل بيانات الصندوق الهندسي (بدون أي حاجة لنص ML Kit)
    data class HybridBox(
        val x: Int,
        val y: Int,
        val width: Int,
        val height: Int,
        val angle: Float
    )

    // ✅ نتيجة جديدة: نص Paddle مع إحداثيات الصندوق المدمج
    data class HybridResult(
        val text: String,
        val box: HybridBox
    )

    override fun recognize(image: InputImage): String {
        val results = recognizeWithBoxes(image)
        return results.joinToString("\n") { it.text }
    }

    // ✅ الدالة الجديدة: تنفذ الخطوات 1-5 وتعيد النصوص مع الإحداثيات
    fun recognizeWithBoxes(image: InputImage): List<HybridResult> {
        val bitmap = image.bitmapInternal ?: return emptyList()
        val results = mutableListOf<HybridResult>()

        try {
            // 1. استلام الصورة وعمل ML Kit OCR لاستخراج الصناديق
            val mlKitResult: Text = mlKitRecognizer.process(image)
            val rawBoxes = mutableListOf<HybridBox>()

            for (block in mlKitResult.textBlocks) {
                val box = block.boundingBox ?: continue
                val line = block.lines.firstOrNull()
                val angle = line?.angle ?: 0f

                rawBoxes.add(
                    HybridBox(
                        x = box.left,
                        y = box.top,
                        width = box.width(),
                        height = box.height(),
                        angle = angle
                    )
                )
            }

            // 2. دمج فقاعات ML Kit هندسياً
            val mergedBoxes = mergeBoxesIntelligently(rawBoxes)

            // 3. استخدام صناديق ML Kit المدمجة لقص أجزاء من الصورة
            // 4. إرسال الأجزاء المقصوصة إلى Paddle OCR
            for (box in mergedBoxes) {
                val startX = max(0, box.x)
                val startY = max(0, box.y)
                val endX = min(bitmap.width, box.x + box.width)
                val endY = min(bitmap.height, box.y + box.height)

                val cropW = endX - startX
                val cropH = endY - startY
                if (cropW < 10 || cropH < 10) continue

                var croppedBitmap: Bitmap? = null
                var rotatedBitmap: Bitmap? = null
                try {
                    // قص مساحة الفقاعة المدمجة بدقة من الصورة الأصلية
                    croppedBitmap = Bitmap.createBitmap(bitmap, startX, startY, cropW, cropH)

                    // تصحيح الدوران بناءً على زاوية الميل
                    rotatedBitmap = if (abs(box.angle) > 1f) {
                        val matrix = Matrix().apply {
                            postRotate(-box.angle, cropW / 2f, cropH / 2f)
                        }
                        Bitmap.createBitmap(croppedBitmap, 0, 0, cropW, cropH, matrix, true)
                    } else {
                        croppedBitmap
                    }

                    val croppedInput = InputImage.fromBitmap(rotatedBitmap, 0)

                    // قراءة النص حصرياً عبر PaddleOCR
                    val paddleText = paddleRecognizer.recognize(croppedInput)

                    // 5. إعادة نص Paddle مع إحداثيات الصندوق
                    if (paddleText.isNotBlank()) {
                        results.add(HybridResult(paddleText.trim(), box))
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "خطأ أثناء معالجة القصة الهجينة للصندوق", e)
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

    // دمج الصناديق والمربعات هندسياً بناءً على تقاطع وتداخل الإحداثيات
    private fun mergeBoxesIntelligently(boxes: List<HybridBox>): List<HybridBox> {
        if (boxes.size <= 1) return boxes
        val sorted = boxes.sortedBy { it.y }
        val result = mutableListOf<HybridBox>()
        val visited = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (visited[i]) continue
            var current = sorted[i]

            for (j in i + 1 until sorted.size) {
                if (visited[j]) continue
                val other = sorted[j]

                val xOverlap = max(0, min(current.x + current.width, other.x + other.width) - max(current.x, other.x))
                val yOverlap = max(0, min(current.y + current.height, other.y + other.height) - max(current.y, other.y))

                // دمج المربعات المتقاربة أو المتداخلة
                if (xOverlap > 0 || yOverlap > 0 || (abs(current.y - other.y) < current.height * 0.8f)) {
                    val minX = min(current.x, other.x)
                    val minY = min(current.y, other.y)
                    val maxX = max(current.x + current.width, other.x + other.width)
                    val maxY = max(current.y + current.height, other.y + other.height)

                    current = HybridBox(
                        x = minX,
                        y = minY,
                        width = maxX - minX,
                        height = maxY - minY,
                        angle = current.angle
                    )
                    visited[j] = true
                }
            }
            result.add(current)
        }
        return result
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
