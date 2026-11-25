package eu.kanade.translation

import android.content.Context
import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import eu.kanade.tachiyomi.ui.reader.colorize.AiConfig
import eu.kanade.translation.model.PageTranslation
import eu.kanade.translation.model.TranslationBlock
import eu.kanade.translation.recognizer.HybridTextRecognizer
import eu.kanade.translation.recognizer.TextRecognizerLanguage
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat

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

        // تحديد ما إذا كان الوضع الحالي هو Hybrid لزيادة معامل السماحية للدمج
        val isHybridMode = (mode == OcrMode.HYBRID)

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
                                symWidth = result.box.symWidth,
                                symHeight = result.box.symHeight,
                                angle = result.box.angle,
                                x = result.box.x.toFloat(),
                                y = result.box.y.toFloat(),
                            )
                        }

                        page.blocks = TextBlockMerger.merge(
                            blocks = rawTranslationBlocks,
                            imgWidth = bitmap.width.toFloat(),
                            imgHeight = bitmap.height.toFloat(),
                            collisionPaddingFactor = paddingFactor,
                            isHybrid = isHybridMode,
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
                    // تمرير مؤشر الـ Hybrid لدالة الفحص والدمج
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

            val newX = block.x - (newWidth - block.width) / 2f
            val newY = block.y - (newHeight - block.height) / 2f

            block.copy(
                text = cleanedText,
                translation = cleanedTranslation,
                width = newWidth.coerceAtMost(imgWidth),
                height = newHeight.coerceAtMost(imgHeight),
                x = newX.coerceIn(0f, imgWidth - newWidth.coerceAtMost(imgWidth)),
                y = newY.coerceIn(0f, imgHeight - newHeight.coerceAtMost(imgHeight)),
            )
        }.toMutableList()

        resolveCollisions(expandedBlocks, imgWidth, imgHeight)
        return expandedBlocks
    }

    private data class PairMoveCandidate(
        val ax: Float,
        val ay: Float,
        val bx: Float,
        val by: Float,
        val otherCollisions: Int,
        val totalDisplacement: Float,
    )

    private fun resolveCollisions(blocks: MutableList<TranslationBlock>, imgWidth: Float, imgHeight: Float) {
        val iterations = 8
        for (step in 0 until iterations) {
            var collisionsResolved = 0
            for (i in blocks.indices) {
                for (j in i + 1 until blocks.size) {
                    val a = blocks[i]
                    val b = blocks[j]
                    if (!isOverlapping(a, b)) continue
                    collisionsResolved++

                    val areaA = max(1f, a.width * a.height)
                    val areaB = max(1f, b.width * b.height)
                    // الكتلة الأصغر تتحمل نسبة أكبر من الإزاحة، والكتلة الأكبر تتحمل نسبة أقل.
                    val invA = 1f / areaA
                    val invB = 1f / areaB
                    val shareA = (invA / (invA + invB)).coerceIn(0.2f, 0.8f)
                    val shareB = 1f - shareA

                    val candidates = buildSeparationCandidates(a, b, shareA, shareB, imgWidth, imgHeight)
                    val best = candidates.minWithOrNull(
                        compareBy<PairMoveCandidate> { it.otherCollisions }
                            .thenBy { it.totalDisplacement },
                    )

                    if (best != null) {
                        blocks[i] = a.copy(x = best.ax, y = best.ay)
                        blocks[j] = b.copy(x = best.bx, y = best.by)
                    }

                    // إذا ظل التصادم قائماً بين a و b (مثلاً بسبب حدود الصورة أو تكدس الكتل)، نقلص حجم الأصغر تدريجياً.
                    if (isOverlapping(blocks[i], blocks[j])) {
                        val smallerIdx = if (areaA <= areaB) i else j
                        val smaller = blocks[smallerIdx]
                        val shrinkFactor = 1.0f / AiConfig.translateCollisionPadding.coerceAtLeast(1.05f)
                        val newW = max(1f, smaller.width * shrinkFactor)
                        val newH = max(1f, smaller.height * shrinkFactor)
                        blocks[smallerIdx] = smaller.copy(
                            width = newW,
                            height = newH,
                            x = (smaller.x + (smaller.width - newW) / 2f)
                                .coerceIn(0f, max(0f, imgWidth - newW)),
                            y = (smaller.y + (smaller.height - newH) / 2f)
                                .coerceIn(0f, max(0f, imgHeight - newH)),
                        )
                    }
                }
            }
            if (collisionsResolved == 0) break
        }
    }

    private fun buildSeparationCandidates(
        a: TranslationBlock,
        b: TranslationBlock,
        shareA: Float,
        shareB: Float,
        imgWidth: Float,
        imgHeight: Float,
        blocks: List<TranslationBlock>? = null,
    ): List<PairMoveCandidate> {
        val cxA = a.x + a.width / 2f
        val cyA = a.y + a.height / 2f
        val cxB = b.x + b.width / 2f
        val cyB = b.y + b.height / 2f

        // مقدار التداخل الأفقي والعمودي بين الكتلتين
        val overlapX = max(0f, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
        val overlapY = max(0f, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))

        // اتجاه الفصل الأساسي يبعد المراكز عن بعضهما (لا يقربهما أبداً)
        val signX = when {
            cxA < cxB -> -1f // a يسار b -> a يذهب لليسار و b يذهب لليمين
            cxA > cxB -> 1f  // a يمين b -> a يذهب لليمين و b يذهب لليسار
            else -> if (a.x <= b.x) -1f else 1f
        }
        val signY = when {
            cyA < cyB -> -1f // a أعلى b -> a يذهب للأعلى و b يذهب للأسفل
            cyA > cyB -> 1f  // a أسفل b -> a يذهب للأسفل و b يذهب للأعلى
            else -> if (a.y <= b.y) -1f else 1f
        }

        val sepX = overlapX + 1f
        val sepY = overlapY + 1f

        val rawPairs = mutableListOf<Pair<Pair<Float, Float>, Pair<Float, Float>>>()

        fun addPair(dAx: Float, dAy: Float, dBx: Float, dBy: Float) {
            rawPairs.add((dAx to dAy) to (dBx to dBy))
        }

        val preferHorizontal = if (overlapX > 0f && overlapY > 0f) {
            overlapX <= overlapY
        } else {
            overlapX > 0f
        }

        if (preferHorizontal) {
            // فصل أفقي موزع بين الكتلتين
            addPair(signX * sepX * shareA, 0f, -signX * sepX * shareB, 0f)
            // فصل عمودي موزع كبديل
            addPair(0f, signY * sepY * shareA, 0f, -signY * sepY * shareB)
            // فصل قطري موزع
            addPair(signX * sepX * shareA * 0.7f, signY * sepY * shareA * 0.7f, -signX * sepX * shareB * 0.7f, -signY * sepY * shareB * 0.7f)
        } else {
            // فصل عمودي موزع بين الكتلتين
            addPair(0f, signY * sepY * shareA, 0f, -signY * sepY * shareB)
            // فصل أفقي موزع كبديل
            addPair(signX * sepX * shareA, 0f, -signX * sepX * shareB, 0f)
            // فصل قطري موزع
            addPair(signX * sepX * shareA * 0.7f, signY * sepY * shareA * 0.7f, -signX * sepX * shareB * 0.7f, -signY * sepY * shareB * 0.7f)
        }

        // إذا كانت إحدى الكتلتين مائلة، نضيف فصلاً على محور زاويتها مع إبعاد المراكز
        val avgAngle = (a.angle + b.angle) / 2f
        if (abs(avgAngle) > 20f && abs(avgAngle) < 160f) {
            val rad = Math.toRadians(avgAngle.toDouble())
            var axisX = cos(rad).toFloat()
            var axisY = sin(rad).toFloat()
            val dot = axisX * (cxA - cxB) + axisY * (cyA - cyB)
            if (dot < 0f) {
                axisX = -axisX
                axisY = -axisY
            }
            val mag = max(sepX, sepY)
            addPair(axisX * mag * shareA, axisY * mag * shareA, -axisX * mag * shareB, -axisY * mag * shareB)
        }

        val maxAX = max(0f, imgWidth - a.width)
        val maxAY = max(0f, imgHeight - a.height)
        val maxBX = max(0f, imgWidth - b.width)
        val maxBY = max(0f, imgHeight - b.height)

        return rawPairs.map { (moveA, moveB) ->
            val nextAX = (a.x + moveA.first).coerceIn(0f, maxAX)
            val nextAY = (a.y + moveA.second).coerceIn(0f, maxAY)
            val nextBX = (b.x + moveB.first).coerceIn(0f, maxBX)
            val nextBY = (b.y + moveB.second).coerceIn(0f, maxBY)

            val movedA = a.copy(x = nextAX, y = nextAY)
            val movedB = b.copy(x = nextBX, y = nextBY)

            var otherCollisions = if (isOverlapping(movedA, movedB)) 1 else 0
            if (blocks != null) {
                for (k in blocks.indices) {
                    val other = blocks[k]
                    if (other === a || other === b) continue
                    if (isOverlapping(movedA, other)) otherCollisions++
                    if (isOverlapping(movedB, other)) otherCollisions++
                }
            }

            val disp = abs(nextAX - a.x) + abs(nextAY - a.y) + abs(nextBX - b.x) + abs(nextBY - b.y)
            PairMoveCandidate(
                ax = nextAX,
                ay = nextAY,
                bx = nextBX,
                by = nextBY,
                otherCollisions = otherCollisions,
                totalDisplacement = disp,
            )
        }
    }

    private fun isOverlapping(a: TranslationBlock, b: TranslationBlock): Boolean {
        if (abs(a.angle) < 1f && abs(b.angle) < 1f) {
            return a.x < b.x + b.width && a.x + a.width > b.x &&
                a.y < b.y + b.height && a.y + a.height > b.y
        }
        return checkOBBOverlap(a, b)
    }

    private fun checkOBBOverlap(a: TranslationBlock, b: TranslationBlock): Boolean {
        val radA = Math.toRadians(a.angle.toDouble())
        val cosA = cos(radA).toFloat()
        val sinA = sin(radA).toFloat()
        val cxA = a.x + a.width / 2f
        val cyA = a.y + a.height / 2f
        val hwA = a.width / 2f
        val hhA = a.height / 2f

        val radB = Math.toRadians(b.angle.toDouble())
        val cosB = cos(radB).toFloat()
        val sinB = sin(radB).toFloat()
        val cxB = b.x + b.width / 2f
        val cyB = b.y + b.height / 2f
        val hwB = b.width / 2f
        val hhB = b.height / 2f

        val dx = cxB - cxA
        val dy = cyB - cyA

        fun testAxis(axisX: Float, axisY: Float): Boolean {
            val distance = abs(dx * axisX + dy * axisY)
            val radiusA = hwA * abs(cosA * axisX + sinA * axisY) + hhA * abs(-sinA * axisX + cosA * axisY)
            val radiusB = hwB * abs(cosB * axisX + sinB * axisY) + hhB * abs(-sinB * axisX + cosB * axisY)
            return distance <= (radiusA + radiusB)
        }

        return testAxis(cosA, sinA) && testAxis(-sinA, cosA) && testAxis(cosB, sinB) && testAxis(-sinB, cosB)
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

        // زيادة معاملات السماحية للمسافات (الفجوات) إذا كان وضع الـ Hybrid مفعلاً
        val gapMultiplier = 1.2f
        val maxAllowedGapX = sW * gapMultiplier // المسافة الأفقية تبقى طبيعية
        val maxAllowedGapY = if (isHybrid) sH * 1.5f else sH * gapMultiplier // زيادة المسافة العمودية فقط عند تفعيل Hybrid

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
                if (isWebtoon) {
                    if (a.x < b.x) listOf(a, b) else listOf(b, a)
                } else {
                    if (a.x > b.x) listOf(a, b) else listOf(b, a)
                }
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
