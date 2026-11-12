package eu.kanade.translation

import android.content.Context
import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import eu.kanade.translation.model.PageTranslation
import eu.kanade.translation.model.TranslationBlock
import eu.kanade.translation.recognizer.HybridTextRecognizer
import eu.kanade.translation.recognizer.TextRecognizerLanguage
import eu.kanade.translation.recognizer.YoloBubbleDetector
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import tachiyomi.core.common.util.system.logcat
import logcat.LogPriority
import eu.kanade.tachiyomi.ui.reader.colorize.AiConfig

enum class OcrMode { MLKIT, PADDLE, HYBRID }

class OcrBox(
    val text: String,
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val symWidth: Float = PageTextExtractor.DEFAULT_SYMBOL,
    val symHeight: Float = PageTextExtractor.DEFAULT_SYMBOL,
    val angle: Float = 0f,
)

class PageTextExtractor(
    private val mode: OcrMode,
    private val isRtl: Boolean,
    private val mlKit: (suspend (InputImage) -> Text)?,
    private val paddle: (suspend (InputImage) -> List<OcrBox>)?,
    private val context: Context? = null,
    private val recognizerLanguage: TextRecognizerLanguage = TextRecognizerLanguage.JAPANESE,
) {

    suspend fun extract(bitmap: Bitmap): PageTranslation? {
        val image = InputImage.fromBitmap(bitmap, 0)
        logcat(LogPriority.ERROR) { ">>> OCR branch selected: mode=$mode <<<" }

        val paddingFactor = AiConfig.translateCollisionPadding
        val page = PageTranslation(imgWidth = bitmap.width.toFloat(), imgHeight = bitmap.height.toFloat())
        val isHybridMode = (mode == OcrMode.HYBRID)

        // 1. الكشف عن الفقاعات باستخدام نموذج YOLO إذا كان التلوين مفعلاً والمحيط متاحاً
        var yoloBubbles: List<YoloBubbleDetector.RectBox> = emptyList()
        if (context != null && AiConfig.colorizeEnabled) {
            try {
                val detector = YoloBubbleDetector(context)
                yoloBubbles = detector.detect(bitmap)
                detector.close()
            } catch (e: Exception) {
                logcat(LogPriority.ERROR, e) { "YOLO Bubble Detection failed, falling back to standard merger" }
            }
        }

        when (mode) {
            OcrMode.MLKIT -> {
                val boxes = mlKitBoxes(image)
                if (boxes.isEmpty()) return null
                page.blocks = TextBlockMerger.merge(
                    blocks = boxes.map { it.toBlock() },
                    imgWidth = bitmap.width.toFloat(),
                    imgHeight = bitmap.height.toFloat(),
                    collisionPaddingFactor = paddingFactor,
                    isHybrid = isHybridMode,
                    yoloBubbles = yoloBubbles
                )
            }
            OcrMode.PADDLE -> {
                val boxes = paddleBoxes(image)
                if (boxes.isEmpty()) return null
                page.blocks = TextBlockMerger.merge(
                    blocks = boxes.map { it.toBlock() },
                    imgWidth = bitmap.width.toFloat(),
                    imgHeight = bitmap.height.toFloat(),
                    collisionPaddingFactor = paddingFactor,
                    isHybrid = isHybridMode,
                    yoloBubbles = yoloBubbles
                )
            }
            OcrMode.HYBRID -> {
                val ctx = context
                if (ctx != null) {
                    try {
                        val hybridRecognizer = HybridTextRecognizer(ctx, recognizerLanguage)
                        val hybridResults = hybridRecognizer.recognizeWithBoxes(image)
                        hybridRecognizer.close()

                        if (hybridResults.isEmpty()) return null

                        val rawTranslationBlocks = hybridResults.map { result ->
                            TranslationBlock(
                                text = result.text,
                                width = result.box.width.toFloat(),
                                height = result.box.height.toFloat(),
                                symWidth = DEFAULT_SYMBOL,
                                symHeight = DEFAULT_SYMBOL,
                                angle = result.box.angle,
                                x = result.box.x.toFloat(),
                                y = result.box.y.toFloat()
                            )
                        }

                        page.blocks = TextBlockMerger.merge(
                            blocks = rawTranslationBlocks,
                            imgWidth = bitmap.width.toFloat(),
                            imgHeight = bitmap.height.toFloat(),
                            collisionPaddingFactor = paddingFactor,
                            isHybrid = isHybridMode,
                            yoloBubbles = yoloBubbles
                        )
                    } catch (e: Exception) {
                        logcat(LogPriority.ERROR, e) { "Hybrid processing failed, falling back to ML Kit" }
                        val boxes = mlKitBoxes(image)
                        if (boxes.isEmpty()) return null
                        page.blocks = TextBlockMerger.merge(
                            blocks = boxes.map { it.toBlock() },
                            imgWidth = bitmap.width.toFloat(),
                            imgHeight = bitmap.height.toFloat(),
                            collisionPaddingFactor = paddingFactor,
                            isHybrid = isHybridMode,
                            yoloBubbles = yoloBubbles
                        )
                    }
                } else {
                    val boxes = mlKitBoxes(image)
                    if (boxes.isEmpty()) return null
                    page.blocks = TextBlockMerger.merge(
                        blocks = boxes.map { it.toBlock() },
                        imgWidth = bitmap.width.toFloat(),
                        imgHeight = bitmap.height.toFloat(),
                        collisionPaddingFactor = paddingFactor,
                        isHybrid = isHybridMode,
                        yoloBubbles = yoloBubbles
                    )
                }
            }
        }

        return page.takeIf { it.blocks.isNotEmpty() }
    }

    private suspend fun mlKitBoxes(image: InputImage): List<OcrBox> {
        val recognize = mlKit ?: return emptyList()
        return recognize(image).textBlocks.mapNotNull { block ->
            val bounds = block.boundingBox ?: return@mapNotNull null
            val line = block.lines.firstOrNull()
            val symbol = line?.elements?.firstOrNull()?.symbols?.firstOrNull()?.boundingBox
            OcrBox(
                text = block.text,
                x = bounds.left.toFloat(),
                y = bounds.top.toFloat(),
                width = bounds.width().toFloat(),
                height = bounds.height().toFloat(),
                symWidth = symbol?.width()?.toFloat() ?: DEFAULT_SYMBOL,
                symHeight = symbol?.height()?.toFloat() ?: DEFAULT_SYMBOL,
                angle = line?.angle ?: 0f,
            )
        }
    }

    private suspend fun paddleBoxes(image: InputImage): List<OcrBox> {
        val recognize = paddle ?: return emptyList()
        return recognize(image).filter { it.text.isNotBlank() }
    }

    private fun OcrBox.toBlock() = TranslationBlock(
        text = text,
        width = width,
        height = height,
        symWidth = symWidth,
        symHeight = symHeight,
        angle = angle,
        x = x,
        y = y,
    )

    companion object {
        const val DEFAULT_SYMBOL = 15f

        fun isRtlLanguage(language: String): Boolean = language.lowercase().let {
            it.contains("ja") || it.contains("ko") || it.contains("zh") || it.contains("ar")
        }
    }
}

internal object TextBlockMerger {
    private val urlPattern = Regex("(?i)(https?://\\S+|www\\.\\S+|\\S+\\.(com|net|org|io|me|cc|tv|info))")
    private val symbolsFilterPattern = Regex("""[_~•°^@#$&|\\-]""")

    fun merge(
        blocks: List<TranslationBlock>, 
        imgWidth: Float, 
        imgHeight: Float, 
        collisionPaddingFactor: Float = 1.3f,
        isHybrid: Boolean = false,
        yoloBubbles: List<YoloBubbleDetector.RectBox> = emptyList()
    ): MutableList<TranslationBlock> {
        if (blocks.isEmpty()) return mutableListOf()

        val filteredBlocks = blocks.mapNotNull { block ->
            val cleanedText = block.text
                .replace(urlPattern, "")
                .replace(symbolsFilterPattern, " ")
                .replace(Regex("\\s+"), " ")
                .trim()

            if (cleanedText.isBlank()) null else block.copy(text = cleanedText)
        }

        if (filteredBlocks.isEmpty()) return mutableListOf()

        val isWebtoon = imgHeight > (imgWidth * 2.0f)
        var currentBlocks = filteredBlocks.toMutableList()
        var mergedAny: Boolean

        do {
            mergedAny = false
            val nextPassBlocks = mutableListOf<TranslationBlock>()
            val visited = BooleanArray(currentBlocks.size) { false }

            for (i in currentBlocks.indices) {
                if (visited[i]) continue
                var currentMerged = currentBlocks[i]

                for (j in i + 1 until currentBlocks.size) {
                    if (visited[j]) continue
                    if (shouldMergeTextBlock(currentMerged, currentBlocks[j], isHybrid)) {
                        currentMerged = mergeTextBlock(currentMerged, currentBlocks[j], isWebtoon)
                        visited[j] = true
                        mergedAny = true
                    }
                }
                nextPassBlocks.add(currentMerged)
            }
            currentBlocks = nextPassBlocks
        } while (mergedAny && currentBlocks.size > 1)

        val mergedBlocks = currentBlocks.mapNotNull { block ->
            if (block.text.trim().isBlank()) null else block
        }.toMutableList()

        if (mergedBlocks.isEmpty()) return mutableListOf()

        val expandedBlocks = mergedBlocks.map { block ->
            val cleanedText = block.text.replace("\n", " ").trim()
            val cleanedTranslation = block.translation?.replace("\n", " ")?.trim() ?: ""

            val textRatio = (cleanedTranslation.length.toFloat() / cleanedText.length.coerceAtLeast(1))
                .coerceIn(1.0f, 1.25f)
            val finalScale = sqrt(textRatio.toDouble()).toFloat()

            var newHeight: Float
            var newWidth: Float
            if (abs(block.angle) in 70.0..110.0) {
                newHeight = block.height * finalScale
                newWidth = block.width * finalScale * collisionPaddingFactor
            } else {
                newHeight = block.height * finalScale * collisionPaddingFactor
                newWidth = block.width * finalScale
            }

            var newX = block.x - (newWidth - block.width) / 2f
            var newY = block.y - (newHeight - block.height) / 2f

            var scaledBlock = block.copy(
                text = cleanedText,
                translation = cleanedTranslation,
                width = newWidth.coerceAtMost(imgWidth),
                height = newHeight.coerceAtMost(imgHeight),
                x = newX.coerceIn(0f, imgWidth - newWidth.coerceAtMost(imgWidth)),
                y = newY.coerceIn(0f, imgHeight - newHeight.coerceAtMost(imgHeight)),
            )

            // دمج إحداثيات YOLO إذا كان التلوين مفعلاً وتم رصد فقاعة تحتضن النص
            if (AiConfig.colorizeEnabled && yoloBubbles.isNotEmpty()) {
                scaledBlock = applyYoloBubbleBounds(scaledBlock, yoloBubbles)
            }

            scaledBlock
        }.toMutableList()

        return expandedBlocks
    }

    /**
     * مطابقة وتوسيع صندوق النص ليطابق فقاعة YOLO التي تتقاطع معه أو تحتويه
     */
    private fun applyYoloBubbleBounds(block: TranslationBlock, bubbles: List<YoloBubbleDetector.RectBox>): TranslationBlock {
        val blockCenterX = block.x + block.width / 2f
        val blockCenterY = block.y + block.height / 2f

        // البحث عن أقرب فقاعة YOLO يحتوي مركزها أو تتقاطع بقوة مع صندوق النص
        val matchingBubble = bubbles.find { bubble ->
            blockCenterX >= bubble.x && blockCenterX <= (bubble.x + bubble.width) &&
            blockCenterY >= bubble.y && blockCenterY <= (bubble.y + bubble.height)
        } ?: bubbles.minByOrNull { bubble ->
            val bCenterX = bubble.x + bubble.width / 2f
            val bCenterY = bubble.y + bubble.height / 2f
            abs(blockCenterX - bCenterX) + abs(blockCenterY - bCenterY)
        }

        // إذا وجدنا فقاعة قريبة ومناسبة، نتبنى إحداثياتها لتغطية الفقاعة بالكامل
        if (matchingBubble != null) {
            val overlapX = min(block.x + block.width, matchingBubble.x + matchingBubble.width) - max(block.x, matchingBubble.x)
            val overlapY = min(block.y + block.height, matchingBubble.y + matchingBubble.height) - max(block.y, matchingBubble.y)
            
            // إذا كان هناك تقاطع أو أن الفقاعة تحيط بالنص
            if (overlapX > 0 || overlapY > 0 || matchingBubble.score > 0.5f) {
                return block.copy(
                    x = matchingBubble.x,
                    y = matchingBubble.y,
                    width = matchingBubble.width,
                    height = matchingBubble.height
                )
            }
        }

        return block
    }

    private fun shouldMergeTextBlock(r1: TranslationBlock, r2: TranslationBlock, isHybrid: Boolean): Boolean {
        val angleDiff = abs(r1.angle - r2.angle)
        val angleSimilar = angleDiff < 15 || abs(angleDiff - 180) < 15
        if (!angleSimilar) return false

        val isVertical = abs(r1.angle) in 70.0..110.0
        val r1Right = r1.x + r1.width
        val r1Bottom = r1.y + r1.height
        val r2Right = r2.x + r2.width
        val r2Bottom = r2.y + r2.height

        val sH = max(r1.symHeight, max(r2.symHeight, 12f))
        val sW = max(r1.symWidth, max(r2.symWidth, 12f))

        val gapMultiplier = 1.2f
        val maxAllowedGapX = sW * gapMultiplier 
        val maxAllowedGapY = if (isHybrid) sH * 1.5f else sH * gapMultiplier 
        
        val yOverlap = max(0f, min(r1Bottom, r2Bottom) - max(r1.y, r2.y))
        val xOverlap = max(0f, min(r1Right, r2Right) - max(r1.x, r2.x))

        val minHeight = min(r1.height, r2.height)
        val isFullVerticalCover = yOverlap >= (minHeight * 0.75f)

        val minWidth = min(r1.width, r2.width)
        val isFullHorizontalCover = xOverlap >= (minWidth * 0.95f)
        val isInCross = isFullHorizontalCover || isFullVerticalCover
        if (!isInCross) return false

        if (isVertical) {
            val sideGap = if (r1.x < r2.x) r2.x - r1Right else r1.x - r2Right
            if (sideGap > maxAllowedGapX) return false
            if (isFullVerticalCover && sideGap <= 0f) return true

            val vertGap = if (r1.y < r2.y) r2.y - r1Bottom else r1.y - r2Bottom
            if (vertGap > maxAllowedGapY) return false

            val isTouchingOrClose = sideGap <= maxAllowedGapX && vertGap <= maxAllowedGapY
            if (!isTouchingOrClose) return false

            val dy = abs(r1.y - r2.y)
            val dx = abs(r1.x - r2.x)
            val isOriginsClose = dy < maxAllowedGapY && dx < maxAllowedGapX
            val isSideBySide = sideGap < maxAllowedGapX && dy < maxAllowedGapY
            val alignedVertically = yOverlap > (sH * 0.15f)
            val closeHorizontally = sideGap < maxAllowedGapX
            return isOriginsClose || isSideBySide || (closeHorizontally && alignedVertically)
        } else {
            val vGap = max(0f, if (r1.y < r2.y) r2.y - r1Bottom else r1.y - r2Bottom)
            if (vGap > maxAllowedGapY) return false
            if (isFullHorizontalCover && vGap <= maxAllowedGapY) return true

            val sideGap = if (r1.x < r2.x) r2.x - r1Right else r1.x - r2Right
            if (sideGap > maxAllowedGapX) return false

            val hasHighSideOverlap = xOverlap > (minWidth * 0.70f)
            val centerR1X = r1.x + r1.width / 2f
            val centerR2X = r2.x + r2.width / 2f
            val centersAligned = abs(centerR1X - centerR2X) < max(r1.width, r2.width) * 0.35f
            val isTouching = vGap <= maxAllowedGapY
            return (hasHighSideOverlap || centersAligned) && isTouching
        }
    }

    private fun mergeTextBlock(a: TranslationBlock, b: TranslationBlock, isWebtoon: Boolean): TranslationBlock {
        val minX = min(a.x, b.x)
        val minY = min(a.y, b.y)
        val maxX = max(a.x + a.width, b.x + b.width)
        val maxY = max(a.y + a.height, b.y + b.height)
        val isVertical = abs(a.angle) in 70.0..110.0
        val ordered = if (isVertical) {
            if (a.x > b.x) listOf(a, b) else listOf(b, a)
        } else {
            if (abs(a.y - b.y) > max(a.symHeight, b.symHeight) * 0.5f) {
                if (a.y < b.y) listOf(a, b) else listOf(b, a)
            } else {
                if (isWebtoon) { if (a.x < b.x) listOf(a, b) else listOf(b, a) }
                else { if (a.x > b.x) listOf(a, b) else listOf(b, a) }
            }
        }

        val totalLen = (a.text.length + b.text.length).coerceAtLeast(1)
        val joinedText = if (ordered.any { it.text.any { c -> c.code in 0x4E00..0x9FFF || c.code in 0x3040..0x309F || c.code in 0x30A0..0x30FF } }) {
            ordered.joinToString("") { it.text.trim() }
        } else {
            ordered.joinToString(" ") { it.text.trim() }
        }

        val joinedTranslation = ordered.joinToString(" ") { it.translation.trim() }.trim()

        return TranslationBlock(
            text = joinedText,
            translation = joinedTranslation,
            width = maxX - minX, 
            height = maxY - minY, 
            x = minX, 
            y = minY,
            angle = if (a.text.length >= b.text.length) a.angle else b.angle,
            symWidth = (a.symWidth * a.text.length + b.symWidth * b.text.length) / totalLen,
            symHeight = (a.symHeight * a.text.length + b.symHeight * b.text.length) / totalLen,
        )
    }
}
