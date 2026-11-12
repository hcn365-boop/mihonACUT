package eu.kanade.translation

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import eu.kanade.translation.model.PageTranslation
import eu.kanade.translation.model.TranslationBlock
import kotlin.math.max
import kotlin.math.min

/**
 * Draws the translation of a page into its bitmap using a two-step strategy:
 * 1. Erases/covers all original text boxes across the entire page first.
 * 2. Draws all translated texts over the cleaned boxes second, with a nice inner padding.
 */
object PageTranslationRenderer {

    private const val MIN_BOX_PX = 6f
    private const val MIN_TEXT_PX = 8
    private const val MAX_TEXT_PX = 200
    private const val BUBBLE_COLOR = Color.WHITE

    /** Blocks whose box is tilted more than this are treated as vertical and drawn straight. */
    private const val VERTICAL_ANGLE = 85f

    fun render(source: Bitmap, page: PageTranslation, typeface: Typeface = Typeface.DEFAULT): Bitmap {
        if (page.imgWidth <= 0f || page.blocks.none { it.translation.isNotBlank() }) return source
        val output = source.copy(Bitmap.Config.ARGB_8888, true) ?: return source
        val canvas = Canvas(output)
        val scale = output.width / page.imgWidth

        val validBlocks = page.blocks.filter { it.translation.isNotBlank() }
        if (validBlocks.isEmpty()) return output

        // ==========================================
        // الخطوة الأولى: مسح وطمس كل الفقاعات في الصفحة أولاً
        // ==========================================
        for (block in validBlocks) {
            eraseOriginalText(output, canvas, block, scale)
        }

        // ==========================================
        // الخطوة الثانية: طباعة كل النصوص المترجمة فوق الفقاعات النظيفة ثانياً
        // ==========================================
        for (block in validBlocks) {
            drawTranslatedText(canvas, block, scale, typeface)
        }

        return output
    }

    /** الخطوة 1: مسح النص الأصلي برسم خلفية نظيفة بلون أبيض ثابت */
    private fun eraseOriginalText(bitmap: Bitmap, canvas: Canvas, block: TranslationBlock, scale: Float) {
        val metrics = calculateBoxCoordinates(bitmap, block, scale)
        if (metrics.width < MIN_BOX_PX || metrics.height < MIN_BOX_PX) return

        val radius = min(metrics.width, metrics.height) * 0.15f
        canvas.drawRoundRect(metrics.left, metrics.top, metrics.right, metrics.bottom, radius, radius, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = BUBBLE_COLOR })
    }

    /** الخطوة 2: حساب حجم النص المناسب وطباعته داخل الإطار مع الـ Padding الجديد (5%) */
    private fun drawTranslatedText(canvas: Canvas, block: TranslationBlock, scale: Float, typeface: Typeface) {
        // حساب الـ Padding بنسبة 1% من أبعاد الصندوق لتجنب التصاق النص بالحواف
        val paddingX = block.width * scale * 0.01f
        val paddingY = block.height * scale * 0.01f

        val left = max(block.x * scale + paddingX, 0f)
        val top = max(block.y * scale + paddingY, 0f)
        val right = max(left + (block.width * scale) - (paddingX * 2), left + 10f)
        val bottom = max(top + (block.height * scale) - (paddingY * 2), top + 10f)
        
        val width = right - left
        val height = bottom - top
        if (width < MIN_BOX_PX || height < MIN_BOX_PX) return

        val paint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
            this.typeface = typeface
            color = Color.BLACK
        }
        
        val layout = fitText(block.translation, paint, width.toInt(), height.toInt())

        canvas.save()
        canvas.translate(left + width / 2, top + height / 2)
        if (block.angle <= VERTICAL_ANGLE) canvas.rotate(block.angle)
        canvas.translate(-layout.width / 2f, -layout.height / 2f)
        layout.draw(canvas)
        canvas.restore()
    }

    private data class BoxMetrics(
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float,
        val width: Float,
        val height: Float
    )

    private fun calculateBoxCoordinates(bitmap: Bitmap, block: TranslationBlock, scale: Float): BoxMetrics {
        // مساحة مسح الخلفية تبقى بحجم الصندوق الأصلي لضمان مسح النص بالكامل
        val padX = block.symWidth * 2
        val padY = block.symHeight
        val left = max((block.x - padX / 2) * scale, 0f)
        val top = max((block.y - padY / 2) * scale, 0f)
        val right = min(left + (block.width + padX) * scale, bitmap.width.toFloat())
        val bottom = min(top + (block.height + padY) * scale, bitmap.height.toFloat())
        val width = right - left
        val height = bottom - top
        return BoxMetrics(left, top, right, bottom, width, height)
    }

    /** 
     * خوارزمية البحث الثنائي الذكية لحجم الخط مع قياس دقيق لعرض الكلمات 
     * لضمان ملاءمة النص داخل الارتفاع والعرض دون كسر الكلمات.
     */
    private fun fitText(text: String, paint: TextPaint, width: Int, height: Int): StaticLayout {
        var low = MIN_TEXT_PX
        var high = max(low, min(height, MAX_TEXT_PX))
        var best = low
        
        var bestLayout: StaticLayout? = null

        while (low <= high) {
            val mid = (low + high) / 2
            paint.textSize = mid.toFloat()
            
            // توزيع النص مسبقاً بناءً على الحجم الحالي للخط وعرض الصندوق
            val formattedText = formatTextToPreventCutting(text, paint, width)
            val currentLayout = layout(formattedText, paint, width)

            if (currentLayout.height <= height) {
                best = mid
                bestLayout = currentLayout
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        
        // ضبط حجم الخط النهائي وإعادة البناء
        paint.textSize = best.toFloat()
        val finalFormattedText = formatTextToPreventCutting(text, paint, width)
        return layout(finalFormattedText, paint, width)
    }

    /**
     * خوارزمية الكسر المسبق للأسطر (Word-Bound Wrapping Algorithm):
     * تقوم بفحص الكلمات بناءً على الفراغات، وإذا تبين أن الكلمة التالية ستقطع حافة 
     * عرض الفقاعة، يتم فصلها مسبقاً بنزول سطر تلقائي (\n) لحمايتها من التقطيع.
     */
    private fun formatTextToPreventCutting(text: String, paint: TextPaint, maxLineWidth: Int): String {
        if (maxLineWidth <= 0) return text
        
        val words = text.split(Regex("(?<=\\s)|(?=\\s)"))
        val lineBuilder = StringBuilder()
        val currentLine = StringBuilder()

        for (word in words) {
            val testLine = currentLine.toString() + word
            // إذا كان عرض السطر مع الكلمة الجديدة يتجاوز عرض الفقاعة المتاح
            if (paint.measureText(testLine) > maxLineWidth) {
                if (currentLine.isNotEmpty()) {
                    lineBuilder.append(currentLine.toString().trim()).append("\n")
                    currentLine.clear()
                }
            }
            currentLine.append(word)
        }
        if (currentLine.isNotEmpty()) {
            lineBuilder.append(currentLine.toString().trim())
        }

        return lineBuilder.toString()
    }

    private fun layout(text: String, paint: TextPaint, width: Int): StaticLayout =
        StaticLayout.Builder.obtain(text, 0, text.length, paint, max(width, 1))
            .setAlignment(Layout.Alignment.ALIGN_CENTER)
            .setIncludePad(false)
            .setBreakStrategy(StaticLayout.BREAK_STRATEGY_BALANCED)
            .build()
}
