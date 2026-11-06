package eu.kanade.translation.recognizer

import android.content.Context
import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import android.util.Log
import kotlin.math.max
import kotlin.math.min

class HybridTextRecognizer(
    private val context: Context,
    override val language: TextRecognizerLanguage
) : TextRecognizer {

    private val TAG = "HybridTextRecognizer"
    
    // نعتمد على ML Kit لاستخراج الصناديق والإحداثيات بدقة فائقة
    private val mlKitRecognizer = MlKitTextRecognizer(language)
    
    // نعتمد على نموذج PaddleOCR للتعرف الفعلي على النصوص (قراءة الحروف)
    private val paddleRecognizer = PaddleTextRecognizer(context, language)

    override fun recognize(image: InputImage): String {
        val bitmap = image.bitmapInternal ?: return mlKitRecognizer.recognize(image)
        
        try {
            // 1. الخطوة الأولى: استخدام ML Kit للحصول على الصناديق (Bounding Boxes) بدقة
            val mlKitResult: Text = mlKitRecognizer.process(image)
            val extractedBlocks = mutableListOf<String>()

            for (block in mlKitResult.textBlocks) {
                val box = block.boundingBox ?: continue
                if (block.text.isBlank()) continue

                val startX = max(0, box.left)
                val startY = max(0, box.top)
                val endX = min(bitmap.width, box.right)
                val endY = min(bitmap.height, box.bottom)

                val cropW = endX - startX
                val cropH = endY - startY
                if (cropW < 10 || cropH < 10) continue

                var croppedBitmap: Bitmap? = null
                try {
                    // 2. قص الجزء الخاص بالنص بدقة بناءً على إحداثيات ML Kit
                    croppedBitmap = Bitmap.createBitmap(bitmap, startX, startY, cropW, cropH)
                    val croppedInput = InputImage.fromBitmap(croppedBitmap, 0)

                    // 3. تمرير الصورة المقصوصة إلى PaddleOCR للتعرف على النص بداخله
                    // (ملاحظة: سنستدعي دالة التعرف المباشرة من Paddle على الكوب المقتص)
                    val paddleText = paddleRecognizer.recognize(croppedInput)

                    if (paddleText.isNotBlank()) {
                        extractedBlocks.add(paddleText.trim())
                    } else {
                        // كخطوة احتياطية: إذا فشل Paddle في قراءة النص المقصوص، نلجأ لنص ML Kit الأصلي
                        extractedBlocks.add(block.text.trim())
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "خطأ أثناء معالجة الصندوق الهجين", e)
                    extractedBlocks.add(block.text.trim())
                } finally {
                    croppedBitmap?.recycle()
                }
            }

            return extractedBlocks.joinToString("\n")
        } catch (e: Exception) {
            Log.e(TAG, "فشل في تشغيل Hybrid OCR، الرجوع إلى ML Kit العادي", e)
            return mlKitRecognizer.recognize(image)
        }
    }

    override fun close() {
        mlKitRecognizer.close()
        paddleRecognizer.close()
    }
}
