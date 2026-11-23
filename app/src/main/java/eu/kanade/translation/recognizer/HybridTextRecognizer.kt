package eu.kanade.translation.recognizer

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import eu.kanade.translation.PageTextExtractor
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
        val angle: Float,
        val symWidth: Float = PageTextExtractor.DEFAULT_SYMBOL,
        val symHeight: Float = PageTextExtractor.DEFAULT_SYMBOL
    )

    data class HybridResult(
        val text: String,
        val box: HybridBox
    )

    override fun recognize(image: InputImage): String {
        val results = recognizeWithBoxes(image)
        return results.joinToString("\n") { it.text }
    }

    // استخراج الصناديق وأبعاد الرموز من ML Kit ثم قراءة النصوص الداخلية عبر PaddleOCR بعد توسيع الصندوق 10%
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

                val symbolBoxes = block.lines
                    .flatMap { it.elements }
                    .flatMap { it.symbols }
                    .mapNotNull { it.boundingBox }

                val symWidth = if (symbolBoxes.isNotEmpty()) {
                    symbolBoxes.map { it.width().toFloat() }.average().toFloat()
                        .coerceAtLeast(4f)
                } else {
                    PageTextExtractor.DEFAULT_SYMBOL
                }

                val symHeight = if (symbolBoxes.isNotEmpty()) {
                    symbolBoxes.map { it.height().toFloat() }.average().toFloat()
                        .coerceAtLeast(4f)
                } else {
                    PageTextExtractor.DEFAULT_SYMBOL
                }

                rawBoxes.add(
                    HybridBox(
                        x = box.left,
                        y = box.top,
                        width = box.width(),
                        height = box.height(),
                        angle = angle,
                        symWidth = symWidth,
                        symHeight = symHeight
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

                // توسيع الصندوق بنسبة 10% لتجنب قص حواف النصوص
                val padX = (cropW * 0.10f).toInt()
                val padY = (cropH * 0.10f).toInt()

                val finalStartX = max(0, startX - padX)
                val finalStartY = max(0, startY - padY)
                val finalEndX = min(bitmap.width, endX + padX)
                val finalEndY = min(bitmap.height, endY + padY)

                val finalCropW = finalEndX - finalStartX
                val finalCropH = finalEndY - finalStartY
                if (finalCropW < 10 || finalCropH < 10) continue

                val adjustedBox = HybridBox(
                    x = finalStartX,
                    y = finalStartY,
                    width = finalCropW,
                    height = finalCropH,
                    angle = box.angle,
                    symWidth = box.symWidth,
                    symHeight = box.symHeight
                )

                var croppedBitmap: Bitmap? = null
                var rotatedBitmap: Bitmap? = null
                try {
                    croppedBitmap = Bitmap.createBitmap(bitmap, finalStartX, finalStartY, finalCropW, finalCropH)

                    rotatedBitmap = if (abs(box.angle) > 1f) {
                        val matrix = Matrix().apply {
                            postRotate(-box.angle, finalCropW / 2f, finalCropH / 2f)
                        }
                        Bitmap.createBitmap(croppedBitmap, 0, 0, finalCropW, finalCropH, matrix, true)
                    } else {
                        croppedBitmap
                    }

                    val croppedInput = InputImage.fromBitmap(rotatedBitmap, 0)
                    val paddleText = paddleRecognizer.recognize(croppedInput)

                    if (paddleText.isNotBlank()) {
                        results.add(HybridResult(paddleText.trim(), adjustedBox))
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
