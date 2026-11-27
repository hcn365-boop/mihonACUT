package eu.kanade.translation

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import eu.kanade.translation.model.PageTranslation
import eu.kanade.translation.model.TranslationBlock
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

object PageTranslationRenderer {

    private const val MIN_BOX_PX = 6f
    private const val MIN_TEXT_PX = 10
    private const val MAX_TEXT_PX = 220
    private const val BUBBLE_COLOR = Color.WHITE

    /** Blocks whose box is tilted more than this are treated as vertical and drawn straight. */
    private const val VERTICAL_ANGLE = 85f

    fun render(
        source: Bitmap,
        page: PageTranslation,
        typeface: Typeface = Typeface.DEFAULT,
        boldText: Boolean = true,
    ): Bitmap {
        if (page.imgWidth <= 0f || page.blocks.none { it.translation.isNotBlank() }) return source
        val output = source.copy(Bitmap.Config.ARGB_8888, true) ?: return source
        val canvas = Canvas(output)
        val scale = output.width / page.imgWidth

        val validBlocks = page.blocks.filter { it.translation.isNotBlank() }
        if (validBlocks.isEmpty()) return output

        val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = BUBBLE_COLOR
        }

        for (block in validBlocks) {
            eraseOriginalText(output, canvas, block, scale, fillPaint)
        }

        for (block in validBlocks) {
            drawTranslatedText(canvas, block, scale, typeface, boldText)
        }

        return output
    }

    private fun eraseOriginalText(
        bitmap: Bitmap,
        canvas: Canvas,
        block: TranslationBlock,
        scale: Float,
        fillPaint: Paint,
    ) {
        val metrics = calculateBoxCoordinates(bitmap, block, scale)
        if (metrics.width < MIN_BOX_PX || metrics.height < MIN_BOX_PX) return

        val rect = RectF(metrics.left, metrics.top, metrics.right, metrics.bottom)
        val radius = min(metrics.width, metrics.height) * 0.48f

        val hasRotation = abs(block.angle) > 0f && abs(block.angle) <= VERTICAL_ANGLE
        if (hasRotation) {
            canvas.save()
            canvas.rotate(block.angle, rect.centerX(), rect.centerY())
            canvas.drawRoundRect(rect, radius, radius, fillPaint)
            canvas.restore()
        } else {
            canvas.drawRoundRect(rect, radius, radius, fillPaint)
        }
    }

    private fun drawTranslatedText(
        canvas: Canvas,
        block: TranslationBlock,
        scale: Float,
        typeface: Typeface,
        boldText: Boolean,
    ) {
        val maxCanvasW = canvas.width.toFloat()
        val maxCanvasH = canvas.height.toFloat()
        if (maxCanvasW < MIN_BOX_PX || maxCanvasH < MIN_BOX_PX) return

        val paddingX = block.width * scale * 0.03f
        val paddingY = block.height * scale * 0.03f

        val left = ((block.x + block.width) * scale - paddingX).coerceAtMost(maxCanvasW)
            .let { rightCandidate ->
                val rawLeft = max(block.x * scale + paddingX, 0f)
                min(rawLeft, max(0f, rightCandidate - MIN_BOX_PX))
            }
        val top = ((block.y + block.height) * scale - paddingY).coerceAtMost(maxCanvasH)
            .let { bottomCandidate ->
                val rawTop = max(block.y * scale + paddingY, 0f)
                min(rawTop, max(0f, bottomCandidate - MIN_BOX_PX))
            }
        val right = min(max((block.x + block.width) * scale - paddingX, left + MIN_BOX_PX), maxCanvasW)
        val bottom = min(max((block.y + block.height) * scale - paddingY, top + MIN_BOX_PX), maxCanvasH)

        val width = right - left
        val height = bottom - top
        if (width < MIN_BOX_PX || height < MIN_BOX_PX) return

        val paint = TextPaint(Paint.ANTI_ALIAS_FLAG or Paint.SUBPIXEL_TEXT_FLAG).apply {
            this.typeface = typeface
            this.isFakeBoldText = boldText
            color = Color.BLACK
        }

        val layout = fitText(block.translation, paint, width.toInt(), height.toInt())

        canvas.save()
        canvas.translate(left + width / 2f, top + height / 2f)
        if (abs(block.angle) <= VERTICAL_ANGLE) canvas.rotate(block.angle)
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
        val height: Float,
    )

    private fun calculateBoxCoordinates(bitmap: Bitmap, block: TranslationBlock, scale: Float): BoxMetrics {
        val padX = block.symWidth * 1.6f
        val padY = block.symHeight * 1.2f
        val left = max((block.x - padX / 2f) * scale, 0f)
        val top = max((block.y - padY / 2f) * scale, 0f)
        val right = min((block.x + block.width + padX / 2f) * scale, bitmap.width.toFloat())
        val bottom = min((block.y + block.height + padY / 2f) * scale, bitmap.height.toFloat())
        val width = max(0f, right - left)
        val height = max(0f, bottom - top)
        return BoxMetrics(left, top, right, bottom, width, height)
    }

    private fun fitText(text: String, paint: TextPaint, width: Int, height: Int): StaticLayout {
        val low = MIN_TEXT_PX
        val high = max(low, min(height, MAX_TEXT_PX))
        var bestSize = low
        var bestLayout: StaticLayout? = null

        var currentLow = low
        var currentHigh = high

        while (currentLow <= currentHigh) {
            val mid = (currentLow + currentHigh) / 2
            paint.textSize = mid.toFloat()

            val formattedText = formatTextToPreventCutting(text, paint, width)
            val currentLayout = layout(formattedText, paint, width)

            val fitsWidth = (0 until currentLayout.lineCount).all { lineIdx ->
                currentLayout.getLineWidth(lineIdx) <= width.toFloat()
            }
            val fitsHeight = currentLayout.height <= height

            if (fitsHeight && fitsWidth) {
                bestSize = mid
                bestLayout = currentLayout
                currentLow = mid + 1
            } else {
                currentHigh = mid - 1
            }
        }

        if (bestLayout != null) {
            paint.textSize = bestSize.toFloat()
            return bestLayout
        }

        paint.textSize = low.toFloat()
        return layout(formatTextToPreventCutting(text, paint, width), paint, width)
    }

    private fun formatTextToPreventCutting(text: String, paint: TextPaint, maxLineWidth: Int): String {
        if (maxLineWidth <= 0) return text

        val normalized = text.replace('\n', ' ').trim()
        val words = normalized.split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.isEmpty()) return ""

        val lineBuilder = StringBuilder(normalized.length)
        val currentLine = StringBuilder()

        for (word in words) {
            val candidate = if (currentLine.isEmpty()) word else "${currentLine} $word"
            if (paint.measureText(candidate) > maxLineWidth && currentLine.isNotEmpty()) {
                lineBuilder.append(currentLine).append('\n')
                currentLine.clear()
                currentLine.append(word)
            } else {
                if (currentLine.isNotEmpty()) currentLine.append(' ')
                currentLine.append(word)
            }
        }
        if (currentLine.isNotEmpty()) {
            lineBuilder.append(currentLine)
        }

        return lineBuilder.toString()
    }

    private fun layout(text: String, paint: TextPaint, width: Int): StaticLayout =
        StaticLayout.Builder.obtain(text, 0, text.length, paint, max(width, 1))
            .setAlignment(Layout.Alignment.ALIGN_CENTER)
            .setIncludePad(false)
            .setLineSpacing(0f, 1.08f)
            .setBreakStrategy(StaticLayout.BREAK_STRATEGY_BALANCED)
            .build()
}
