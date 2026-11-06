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
    
    // محرك ML Kit لاستخراج الصناديق، الإحداثيات وزوايا الميل بدقة عالية
    private val mlKitRecognizer = MlKitTextRecognizer(language)
    
    // محرك PaddleOCR للتعرف الفعلي على محتوى الحروف داخل الصناديق المقصوصة
    private val paddleRecognizer = PaddleTextRecognizer(context, language)

    override fun recognize(image: InputImage): String {
        val bitmap = image.bitmapInternal ?: return mlKitRecognizer.recognize(image)
        
        try {
            // 1. استخراج الصناديق والمواقع باستخدام ML Kit
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
                    // 2. قص مساحة الصندوق بدقة من الصورة الأصلية
                    croppedBitmap = Bitmap.createBitmap(bitmap, startX, startY, cropW, cropH)
                    val croppedInput = InputImage.fromBitmap(croppedBitmap, 0)

                    // 3. محاولة قراءة النص المقصوص بواسطة نموذج PaddleOCR
                    val paddleText = paddleRecognizer.recognize(croppedInput)

                    if (paddleText.isNotBlank()) {
                        extractedBlocks.add(paddleText.trim())
                    } else {
                        // العودة للنص الأصلي لـ ML Kit في حال لم يتعرف Paddle على النص
                        extractedBlocks.add(block.text.trim())
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "خطأ أثناء معالجة القصه الهجينة للصندوق", e)
                    extractedBlocks.add(block.text.trim())
                } finally {
                    croppedBitmap?.recycle()
                }
            }

            return extractedBlocks.joinToString("\n")
        } catch (e: Exception) {
            Log.e(TAG, "فشل في تشغيل النظام الهجين، الرجوع إلى ML Kit", e)
            return mlKitRecognizer.recognize(image)
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
