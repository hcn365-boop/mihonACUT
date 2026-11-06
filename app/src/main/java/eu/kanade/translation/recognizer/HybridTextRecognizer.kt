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

    // استخراج الصناديق بدقة باستخدام ML Kit ثم قراءة النصوص الداخلية عبر PaddleOCR
    fun recognizeWithBoxes(image: InputImage): List<HybridResult> {
        val bitmap = image.bitmapInternal ?: return emptyList()
        val results = mutableListOf<HybridResult>()

        try {
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

            // معالجة القص والتدوير لكل صندوق وقراءته بواسطة PaddleOCR مباشرة
            for (box in rawBoxes) {
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
                    croppedBitmap = Bitmap.createBitmap(bitmap, startX, startY, cropW, cropH)

                    rotatedBitmap = if (abs(box.angle) > 1f) {
                        val matrix = Matrix().apply {
                            postRotate(-box.angle, cropW / 2f, cropH / 2f)
                        }
                        Bitmap.createBitmap(croppedBitmap, 0, 0, cropW, cropH, matrix, true)
                    } else {
                        croppedBitmap
                    }

                    val croppedInput = InputImage.fromBitmap(rotatedBitmap, 0)
                    val paddleText = paddleRecognizer.recognize(croppedInput)

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

    override fun close() {
        try {
            mlKitRecognizer.close()
            paddleRecognizer.close()
        } catch (e: Exception) {
            Log.e(TAG, "خطأ أثناء إغلاق محركات HybridTextRecognizer", e)
        }
    }
}
