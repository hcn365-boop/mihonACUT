package eu.kanade.tachiyomi.ui.reader.colorize

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.os.SystemClock
import eu.kanade.translation.MangaTranslationContext
import eu.kanade.translation.PageTranslationRenderer
import eu.kanade.translation.PageTranslationService
import eu.kanade.translation.TranslationStore
import eu.kanade.translation.model.PageTranslation
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import logcat.LogPriority
import okio.Buffer
import okio.BufferedSource
import tachiyomi.core.common.util.system.ImageUtil
import tachiyomi.core.common.util.system.logcat
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.coroutines.cancellation.CancellationException
import kotlin.math.roundToInt

/**
 * Pipeline order:
 * 1. Upscale/shrink (uncolored) -> 2. OCR & Translate (parallel, does not block colorization) -> 3. Colorize,
 * then draw translated text once colorization finishes (awaiting translation only if still running).
 */
object PageColorizer {

    private const val MODEL_DIR = "colorizer"
    private const val COLORIZER_FILE = "v6_generator.onnx"
    private const val UPSCALER_FILE = "anime4k_acnet.onnx"
    private const val CHROMA_CACHE_DIR = "colorizer_cache"
    private const val PAGE_CACHE_DIR = "ai_pages"

    private const val CHROMA_SIZE = 512 * 512
    private const val MAX_CHROMA_CACHE_BYTES = 256L * 1024 * 1024
    private const val JPEG_QUALITY = 95

    // Separate locks so upscaling, active colorization, and background prefetch do not block each other unnecessarily.
    private val upscaleMutex = Mutex()
    private val colorizeMutex = Mutex()
    private val prefetchChromaMutex = Mutex()

    private class EngineManager<T : Closeable>(private val maxEntries: Int = 2) {
        private val sessions = object : LinkedHashMap<String, T>(16, 0.75f, true) {
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, T>): Boolean {
                if (size > maxEntries) {
                    try {
                        eldest.value.close()
                    } catch (_: Throwable) {}
                    return true
                }
                return false
            }
        }

        fun getOrPut(key: String, factory: () -> T): T {
            synchronized(sessions) {
                sessions[key]?.let { return it }
                val instance = factory()
                sessions[key] = instance
                while (sessions.size > maxEntries) {
                    val iterator = sessions.entries.iterator()
                    if (iterator.hasNext()) {
                        val entry = iterator.next()
                        try {
                            entry.value.close()
                        } catch (_: Throwable) {}
                        iterator.remove()
                    }
                }
                return instance
            }
        }

        fun reset() {
            synchronized(sessions) {
                sessions.values.forEach {
                    try { it.close() } catch (_: Throwable) {}
                }
                sessions.clear()
            }
        }
    }

    private val colorizerManager = EngineManager<MangaColorizer>(maxEntries = 1)
    private val upscalerManager = EngineManager<PageUpscaler>(maxEntries = 1)
    private val acnetManager = EngineManager<Anime4kUpscaler>(maxEntries = 1)

    fun modelFile(context: Context): File = File(File(context.filesDir, MODEL_DIR), COLORIZER_FILE)

    fun upscalerModelFile(context: Context): File = File(File(context.filesDir, MODEL_DIR), UPSCALER_FILE)

    private fun samActive(context: Context): Boolean = AiConfig.colorizeUseSam

    fun isModelInstalled(context: Context): Boolean = modelFile(context).length() > 0

    private fun acnetReady(context: Context): Boolean = true

    fun releaseEngines() {
        logcat(LogPriority.DEBUG) { "[Perf] Releasing AI engines" }
        colorizerManager.reset()
        upscalerManager.reset()
        acnetManager.reset()
    }

    suspend fun colorize(context: Context, source: BufferedSource, intensity: Float): BufferedSource =
        process(context, source)

    suspend fun process(context: Context, source: BufferedSource): BufferedSource {
        return try {
            run(context.applicationContext, source, display = true) ?: source
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e)
            source
        }
    }

    suspend fun processToBytes(context: Context, source: BufferedSource): ByteArray? {
        val result = run(context.applicationContext, source, display = true) ?: return null
        return result.readByteArray()
    }

    suspend fun processImagesToBytes(context: Context, source: BufferedSource): ByteArray? {
        val result = run(context.applicationContext, source, display = false, translate = false) ?: return null
        return result.readByteArray()
    }

    fun contentHash(bytes: ByteArray): String = hash(bytes)

    suspend fun prefetch(context: Context, source: BufferedSource, full: Boolean, translate: Boolean = false) {
        try {
            val app = context.applicationContext
            logcat(LogPriority.DEBUG) { "[Perf] START -> prefetch(full=$full, translate=$translate)" }
            val startMs = SystemClock.elapsedRealtime()
            if (full) run(app, source, display = false, translate = translate) else warmChroma(app, source)
            logcat(LogPriority.DEBUG) { "[Perf] END   -> prefetch(full=$full) took ${SystemClock.elapsedRealtime() - startMs}ms" }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e)
        }
    }

    private inline fun <T> logStep(name: String, block: () -> T): T {
        val startMs = SystemClock.elapsedRealtime()
        logcat(LogPriority.DEBUG) { "[Perf] START -> $name" }
        return try {
            block()
        } finally {
            val elapsedMs = SystemClock.elapsedRealtime() - startMs
            logcat(LogPriority.DEBUG) { "[Perf] END   -> $name (${elapsedMs}ms)" }
        }
    }

    private suspend fun run(
        context: Context,
        source: BufferedSource,
        display: Boolean,
        translate: Boolean = true,
    ): BufferedSource? = coroutineScope {
        val totalStartMs = SystemClock.elapsedRealtime()
        logcat(LogPriority.DEBUG) { "[Perf] START -> PageColorizer.run(display=$display, translate=$translate)" }

        try {
            if (display && !AiConfig.isPost && AiConfig.usePostProcessed() && PostProcessStore.maybeHasResults) {
                val original = logStep("postProcess.readBytes") { source.peek().readByteArray() }
                val originalHash = logStep("postProcess.hash") { hash(original) }
                PostProcessStore.lookup(context, originalHash)?.let { found ->
                    logcat(LogPriority.DEBUG) { "[Perf] CACHE HIT -> PostProcessStore(hash=$originalHash)" }
                    return@coroutineScope logStep("postProcess.compose(hash=$originalHash)") {
                        PostRead.compose(context, originalHash, original, found)
                    }
                }
            }

            val colorModel = modelFile(context)
            val intensity = AiConfig.colorizeIntensity
            val wantColor = AiConfig.colorizeEnabled && intensity > 0f && colorModel.length() > 0
            val wantUpscale = AiConfig.upscaleEnabled && upscaleReady(context)
            val wantFinal = AiConfig.upscaleFinalPassEnabled && acnetReady(context)
            val translation = if (translate) AiConfig.translation else null
            val wantShrink = AiConfig.normalizeSide > 0
            if (!wantColor && !wantUpscale && !wantShrink && !wantFinal && translation == null) {
                logcat(LogPriority.DEBUG) { "[Perf] SKIP -> No AI/translation operations enabled" }
                return@coroutineScope null
            }
            if (ImageUtil.isAnimatedAndSupported(source)) {
                logcat(LogPriority.DEBUG) { "[Perf] SKIP -> Animated image not supported" }
                return@coroutineScope null
            }

            val bytes = logStep("readSourceBytes") { source.peek().readByteArray() }
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            logStep("decodeBounds") {
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
            }
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
                logcat(LogPriority.DEBUG) { "[Perf] SKIP -> Invalid image dimensions (${bounds.outWidth}x${bounds.outHeight})" }
                return@coroutineScope null
            }
            val pixels = bounds.outWidth.toLong() * bounds.outHeight

            val doColor = wantColor && pixels <= AiConfig.colorizeMaxPixels
            val shortSide = minOf(bounds.outWidth, bounds.outHeight)
            val normalize = AiConfig.normalizeSide
            val upscaleInputPixels = if (normalize in 1 until shortSide) {
                val ratio = normalize.toDouble() / shortSide
                (pixels * ratio * ratio).toLong()
            } else {
                pixels
            }
            val doUpscale = wantUpscale && upscaleInputPixels <= AiConfig.upscaleMaxPixels
            val doShrink = normalize in 1 until shortSide
            val translationConfig = translation?.takeIf { pixels <= AiConfig.colorizeMaxPixels }
            if (!doColor && !doUpscale && !doShrink && !wantFinal && translationConfig == null) {
                logcat(LogPriority.DEBUG) { "[Perf] SKIP -> Image exceeds pixel limits for enabled operations (${bounds.outWidth}x${bounds.outHeight})" }
                return@coroutineScope null
            }

            val imageHash = logStep("hashImage") { hash(bytes) }
            val shrinkSide = if (doShrink) normalize else 0
            val baseName = baseName(context, imageHash, if (doColor) intensity else -1f, doUpscale, shrinkSide, colorModel, samActive(context))
            val effective = translationConfig?.effectiveSettings()
            val finalName = translationConfig?.let { config ->
                val s = effective!!
                val key = TranslationStore.key(
                    s.from.name, s.to.name, s.engine.name, s.model, config.ocrMode.name, config.font.name,
                    config.filter.signature, AiConfig.forceTranslate(MangaTranslationContext.mangaId).toString(),
                )
                "${baseName}_T$key"
            }

            if (display && finalName != null) {
                val cached = logStep("readCachedPage(name=$finalName)") {
                    readCachedPage(cacheFile(context, finalName))
                }
                if (cached != null) {
                    logcat(LogPriority.DEBUG) { "[Perf] CACHE HIT -> finalPage(name=$finalName)" }
                    return@coroutineScope cached
                }
            }

            // 1. Upscale/shrink first (on uncolored page), keeping Bitmap in memory when needed downstream.
            val upscaledBitmap: Bitmap? = if (doUpscale || doShrink || wantFinal) {
                logStep("1.upscaleFirst(hash=$imageHash, w=${bounds.outWidth}, h=${bounds.outHeight}, upscale=$doUpscale, shrink=$shrinkSide, final=$wantFinal)") {
                    upscaleFirstBitmap(context, bytes, imageHash, doUpscale, shrinkSide, pixels, wantFinal)
                }
            } else {
                null
            }

            var currentBitmap: Bitmap? = upscaledBitmap
            var ownsCurrentBitmap = upscaledBitmap != null

            try {
                // Helper to obtain a Bitmap for OCR/Colorization without re-decoding when upscaledBitmap is already in memory.
                fun ensureWorkingBitmap(stepName: String): Bitmap? {
                    if (currentBitmap != null) return currentBitmap
                    val decoded = logStep(stepName) { decode(bytes) }
                    if (decoded != null) {
                        currentBitmap = decoded
                        ownsCurrentBitmap = true
                    }
                    return decoded
                }

                // 2. OCR & Translation (starts in parallel without blocking colorization).
                val translationDeferred = if (translationConfig != null && display) {
                    async {
                        logStep("2.translateNow(hash=$imageHash, mode=${translationConfig.ocrMode})") {
                            var ocrBitmap: Bitmap? = null
                            var ownsOcrBitmap = false
                            try {
                                PageTranslationService.translateNow(
                                    context,
                                    effective!!,
                                    translationConfig.ocrMode,
                                    imageHash,
                                    translationConfig.limits,
                                ) {
                                    logStep("2a.decodeForOcr(hash=$imageHash)") {
                                        val existing = currentBitmap
                                        if (existing != null && !existing.isRecycled) {
                                            existing.copy(existing.config ?: Bitmap.Config.ARGB_8888, false).also {
                                                ocrBitmap = it
                                                ownsOcrBitmap = true
                                            }
                                        } else {
                                            decode(bytes).also {
                                                ocrBitmap = it
                                                ownsOcrBitmap = true
                                            }
                                        }
                                    }
                                }
                            } catch (e: CancellationException) {
                                throw e
                            } catch (e: Throwable) {
                                logcat(LogPriority.ERROR, e)
                                null
                            } finally {
                                if (ownsOcrBitmap) ocrBitmap?.recycle()
                            }
                        }
                    }
                } else {
                    if (translationConfig != null) {
                        logStep("2.enqueueTranslation(hash=$imageHash, mode=${translationConfig.ocrMode})") {
                            var ocrBitmap: Bitmap? = null
                            var ownsOcrBitmap = false
                            try {
                                PageTranslationService.enqueue(
                                    context,
                                    effective!!,
                                    translationConfig.ocrMode,
                                    imageHash,
                                    translationConfig.limits,
                                ) {
                                    logStep("2a.decodeForEnqueueOcr(hash=$imageHash)") {
                                        val existing = currentBitmap
                                        if (existing != null && !existing.isRecycled) {
                                            existing.copy(existing.config ?: Bitmap.Config.ARGB_8888, false).also {
                                                ocrBitmap = it
                                                ownsOcrBitmap = true
                                            }
                                        } else {
                                            decode(bytes).also {
                                                ocrBitmap = it
                                                ownsOcrBitmap = true
                                            }
                                        }
                                    }
                                }
                            } finally {
                                if (ownsOcrBitmap) ocrBitmap?.recycle()
                            }
                        }
                    }
                    null
                }

                // 3. Colorize immediately without waiting for translation.
                var coloredBytes: ByteArray? = null
                var coloredBitmap: Bitmap? = null
                if (doColor) {
                    val srcBmp = ensureWorkingBitmap("3b.decodeSource(hash=$imageHash)")
                    if (srcBmp != null) {
                        val colorResult = logStep("3.colorizeOnly(hash=$imageHash, intensity=$intensity)") {
                            colorizeOnlyBitmap(context, srcBmp, imageHash, baseName, intensity, pixels)
                        }
                        if (colorResult != null) {
                            coloredBitmap = colorResult.bitmap
                            coloredBytes = colorResult.encodedBytes
                        }
                    }
                }

                try {
                    val activeBitmap = coloredBitmap ?: currentBitmap
                    if (translationConfig == null || translationDeferred == null) {
                        return@coroutineScope when {
                            coloredBytes != null -> Buffer().write(coloredBytes)
                            activeBitmap != null -> Buffer().write(encode(activeBitmap, activeBitmap.hasAlpha()))
                            else -> null
                        }
                    }

                    // Apply translation after colorization finishes (awaits only if translation is still running).
                    val translatedPage = logStep("4.awaitTranslation(hash=$imageHash)") {
                        translationDeferred.await()
                    } ?: return@coroutineScope when {
                        coloredBytes != null -> Buffer().write(coloredBytes)
                        activeBitmap != null -> Buffer().write(encode(activeBitmap, activeBitmap.hasAlpha()))
                        else -> Buffer().write(bytes)
                    }

                    return@coroutineScope logStep("5.drawTranslation(hash=$imageHash)") {
                        drawTranslationFromBitmap(
                            context = context,
                            config = translationConfig,
                            translatedPage = translatedPage,
                            targetBitmap = activeBitmap,
                            fallbackBytes = coloredBytes ?: bytes,
                            finalName = finalName!!,
                        )
                    } ?: when {
                        coloredBytes != null -> Buffer().write(coloredBytes)
                        activeBitmap != null -> Buffer().write(encode(activeBitmap, activeBitmap.hasAlpha()))
                        else -> Buffer().write(bytes)
                    }
                } finally {
                    coloredBitmap?.recycle()
                }
            } finally {
                if (ownsCurrentBitmap) {
                    currentBitmap?.recycle()
                }
            }
        } finally {
            logcat(LogPriority.DEBUG) { "[Perf] END   -> PageColorizer.run(display=$display) total=${SystemClock.elapsedRealtime() - totalStartMs}ms" }
        }
    }

    /** Step 1: Upscale/refine uncolored image before OCR and colorization, returning Bitmap directly. */
    private suspend fun upscaleFirstBitmap(
        context: Context,
        bytes: ByteArray,
        imageHash: String,
        doUpscale: Boolean,
        shrinkSide: Int,
        pixels: Long,
        wantFinal: Boolean,
    ): Bitmap? {
        val original = logStep("1a.decodeOriginal(hash=$imageHash)") { decode(bytes) } ?: return null
        val working = if (shrinkSide > 0) {
            logStep("1b.shrink(hash=$imageHash, side=$shrinkSide)") { PageResizer.shrink(original, shrinkSide) }
        } else {
            original
        }
        var current: Bitmap = working
        try {
            val upscale = doUpscale && !original.hasAlpha()
            val finalPass = wantFinal && acnetReady(context) && !original.hasAlpha()
            if (!upscale && !finalPass) {
                if (working !== original) {
                    original.recycle()
                    return working
                }
                original.recycle()
                return null
            }
            if (estimateBytes(pixels, color = false, upscale = upscale, scale = upscaleScale()) > AiConfig.memoryBudgetBytes) {
                logcat(LogPriority.DEBUG) { "[Perf] 1.upscaleFirst memory budget exceeded -> returning shrunk/original" }
                if (working !== original) {
                    original.recycle()
                    return working
                }
                original.recycle()
                return null
            }

            logStep("1d.upscaleMutexLock(hash=$imageHash)") {
                upscaleMutex.withLock {
                    if (upscale) {
                        val enlarged = logStep("1e.upscalerEngine.process(hash=$imageHash)") {
                            (upscalerEngine(context) ?: error("No upscaler model"))
                                .process(current, AiConfig.upscaleMaxOutputPixels, PageResizer.MAX_SIDE)
                        }
                        if (current !== working) current.recycle()
                        current = enlarged
                    }
                    if (finalPass) {
                        val refined = logStep("1f.acnetFinalPass(hash=$imageHash)") {
                            acnetFinalPass(context, current)
                        }
                        if (refined !== current && current !== working) current.recycle()
                        current = refined
                    }
                }
            }

            if (current === original) {
                original.recycle()
                return null
            }
            if (working !== original && current !== working) {
                working.recycle()
            }
            original.recycle()
            return current
        } catch (e: CancellationException) {
            if (current !== working && current !== original) current.recycle()
            if (working !== original) working.recycle()
            original.recycle()
            throw e
        } catch (e: Throwable) {
            if (current !== working && current !== original) current.recycle()
            if (working !== original) working.recycle()
            original.recycle()
            val accelerated = doUpscale && AiConfig.upscaleAcceleration != Acceleration.CPU
            if (accelerated && !acceleratorFailed) {
                acceleratorFailed = true
                upscaleMutex.withLock { releaseEngines() }
            }
            throw e
        }
    }

    private class ColorizeResult(val bitmap: Bitmap, val encodedBytes: ByteArray)

    /** Step 3: Colorize (and apply saturation if configured) without waiting for translation. */
    private suspend fun colorizeOnlyBitmap(
        context: Context,
        sourceBitmap: Bitmap,
        imageHash: String,
        baseName: String,
        intensity: Float,
        pixels: Long,
    ): ColorizeResult? {
        val file = cacheFile(context, baseName)
        if (file.isFile && file.length() > 0L) {
            file.setLastModified(System.currentTimeMillis())
            logcat(LogPriority.DEBUG) { "[Perf] 3.colorizeOnly CACHE HIT (name=$baseName)" }
            val cachedBytes = logStep("3a.readColorCache(name=$baseName)") { file.readBytes() }
            val cachedBitmap = decode(cachedBytes)
            if (cachedBitmap != null) {
                return ColorizeResult(cachedBitmap, cachedBytes)
            }
        }

        val grayscale = logStep("3c.looksGrayscale(hash=$imageHash)") { looksGrayscale(sourceBitmap) }
        if (!grayscale) {
            logcat(LogPriority.DEBUG) { "[Perf] 3.colorizeOnly SKIP -> Image is not grayscale" }
            return null
        }
        if (estimateBytes(pixels, color = true, upscale = false, scale = 1) > AiConfig.memoryBudgetBytes) {
            logcat(LogPriority.DEBUG) { "[Perf] 3.colorizeOnly SKIP -> Memory budget exceeded" }
            return null
        }

        val processed = try {
            logStep("3d.colorizeMutexLock(hash=$imageHash)") {
                colorizeMutex.withLock {
                    val colorModel = modelFile(context)
                    val chromaFile = chromaFile(context, imageHash, colorModel)
                    val engine = logStep("3e.colorizerEngine(hash=$imageHash)") {
                        colorizerEngine(context, colorModel)
                    }
                    val chroma = logStep("3f.loadOrPredictChroma(hash=$imageHash)") {
                        loadChroma(chromaFile) ?: engine.predict(sourceBitmap).also {
                            logStep("3g.saveChroma(hash=$imageHash)") {
                                runCatching { saveChroma(context, chromaFile, it) }
                            }
                        }
                    }
                    var page = logStep("3h.engine.render(hash=$imageHash)") {
                        engine.render(sourceBitmap, chroma, intensity)
                    }

                    val saturation = AiConfig.saturation
                    if (saturation != 1f) {
                        val saturated = logStep("3i.applySaturation(hash=$imageHash, sat=$saturation)") {
                            applySaturation(page, saturation)
                        }
                        if (saturated !== page) page.recycle()
                        page = saturated
                    }
                    page
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            if (AiConfig.colorizeAcceleration != Acceleration.CPU && !acceleratorFailed) {
                acceleratorFailed = true
                colorizeMutex.withLock { releaseEngines() }
            }
            throw e
        }

        val encoded = logStep("3j.encodeColored(hash=$imageHash)") { encode(processed, sourceBitmap.hasAlpha()) }
        logStep("3k.saveColorCache(name=$baseName)") {
            runCatching { saveCachedPage(file, encoded) }
        }
        return ColorizeResult(processed, encoded)
    }

    private fun capSize(bitmap: Bitmap): Bitmap = PageResizer.fitTo(bitmap, AiConfig.upscaleMaxOutputPixels)

    private fun acnetFinalPass(context: Context, page: Bitmap): Bitmap {
        val pixels = page.width.toLong() * page.height
        if (pixels > AiConfig.upscaleFinalPassMaxPixels) return page
        if (pixels * (14 + 4 * 21) > AiConfig.memoryBudgetBytes) return page
        val acceleration = if (acceleratorFailed) Acceleration.CPU else AiConfig.upscaleAcceleration
        val key = "${acceleration}_${AiConfig.aiThreads}"

        val engine = acnetManager.getOrPut(key) { Anime4kUpscaler(context, acceleration, AiConfig.aiThreads) }

        val enlarged = logStep("1f1.acnet.upscale(${page.width}x${page.height})") {
            engine.upscale(page)
        }
        if (!AiConfig.upscaleFinalPassKeepSize) return capSize(enlarged)
        val restored = logStep("1f2.acnet.resizeTo(${page.width}x${page.height})") {
            PageResizer.resizeTo(enlarged, page.width, page.height)
        }
        if (restored !== enlarged) enlarged.recycle()
        return restored
    }

    private fun applySaturation(bitmap: Bitmap, saturation: Float): Bitmap {
        val output = Bitmap.createBitmap(bitmap.width, bitmap.height, bitmap.config ?: Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        val paint = Paint().apply {
            colorFilter = ColorMatrixColorFilter(
                ColorMatrix().apply { setSaturation(saturation) }
            )
        }
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return output
    }

    private fun drawTranslationFromBitmap(
        context: Context,
        config: TranslationConfig,
        translatedPage: PageTranslation,
        targetBitmap: Bitmap?,
        fallbackBytes: ByteArray,
        finalName: String,
    ): BufferedSource? {
        val decodedFallback = if (targetBitmap == null || targetBitmap.isRecycled) {
            logStep("5a.decodeTarget") { decode(fallbackBytes) } ?: return null
        } else {
            null
        }
        val target = targetBitmap ?: decodedFallback!!
        var drawn: Bitmap? = null
        try {
            val result = logStep("5b.renderer.render(blocks=${translatedPage.blocks.size})") {
                PageTranslationRenderer.render(target, translatedPage, config.font.typeface(context))
            }
            if (result !== target) drawn = result
            val encoded = logStep("5c.encodeTranslated") { encode(result, target.hasAlpha()) }
            logStep("5d.saveFinalCache(name=$finalName)") {
                runCatching { saveCachedPage(cacheFile(context, finalName), encoded) }
            }
            return Buffer().write(encoded)
        } finally {
            drawn?.recycle()
            decodedFallback?.recycle()
        }
    }

    private suspend fun warmChroma(context: Context, source: BufferedSource) {
        val model = modelFile(context)
        if (!AiConfig.colorizeEnabled || model.length() == 0L) return
        if (ImageUtil.isAnimatedAndSupported(source)) return
        val bytes = source.peek().readByteArray()
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return
        val pixels = bounds.outWidth.toLong() * bounds.outHeight
        if (pixels > AiConfig.colorizeMaxPixels) return
        if (estimateBytes(pixels, true, false, 1) > AiConfig.memoryBudgetBytes) return
        val imageHash = hash(bytes)
        val chromaFile = chromaFile(context, imageHash, model)
        if (chromaFile.isFile) return
        val original = logStep("warmChroma.decode(hash=$imageHash)") { decode(bytes) } ?: return
        try {
            if (!looksGrayscale(original)) return
            prefetchChromaMutex.withLock {
                colorizeMutex.withLock {
                    if (!chromaFile.isFile) {
                        val chroma = logStep("warmChroma.predict(hash=$imageHash)") {
                            colorizerEngine(context, model).predict(original)
                        }
                        logStep("warmChroma.save(hash=$imageHash)") {
                            saveChroma(context, chromaFile, chroma)
                        }
                    }
                }
            }
        } finally {
            original.recycle()
        }
    }

    private fun estimateBytes(pixels: Long, color: Boolean, upscale: Boolean, scale: Int): Long {
        var total = pixels * 4
        if (color) total += pixels * 20
        if (upscale) total += pixels * 14 + pixels * scale * scale * 21
        return total
    }

    private fun decode(bytes: ByteArray): Bitmap? = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

    private fun encode(bitmap: Bitmap, hasAlpha: Boolean): ByteArray {
        val format = if (hasAlpha) Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG
        val output = ByteArrayOutputStream()
        bitmap.compress(format, JPEG_QUALITY, output)
        return output.toByteArray()
    }

    @Volatile
    private var acceleratorFailed = false

    private fun colorizerEngine(context: Context, model: File): MangaColorizer {
        val acceleration = if (acceleratorFailed) Acceleration.CPU else AiConfig.colorizeAcceleration
        val sam = samActive(context)
        val setup = "$acceleration/${AiConfig.aiThreads}/sam=$sam"
        val key = "${model.absolutePath}_${setup}_${model.lastModified()}"
        return colorizerManager.getOrPut(key) {
            logcat(LogPriority.DEBUG) { "[Perf] Creating MangaColorizer engine ($setup)" }
            val guide = if (sam) {
                try {
                    SamFeatureProvider.load(context)
                } catch (e: Throwable) {
                    logcat(LogPriority.ERROR, e) { "SAM features: failed to load binary file" }
                    null
                }
            } else {
                null
            }
            MangaColorizer(model, acceleration, AiConfig.aiThreads, guide)
        }
    }

    private fun chosenModelFiles(context: Context): List<File> =
        listOf(AiConfig.upscaleRestoreModel, AiConfig.upscaleMainModel)
            .filter { it.isNotEmpty() }
            .map { Anime4kModels.file(context, it) }

    private fun upscaleReady(context: Context): Boolean =
        chosenModelFiles(context).let { files -> files.isNotEmpty() && files.all { it.length() > 0 } }

    private fun upscaleScale(): Int =
        Anime4kModels.scaleOf(AiConfig.upscaleRestoreModel) * Anime4kModels.scaleOf(AiConfig.upscaleMainModel)

    private fun upscaleSignature(): String = "m_${AiConfig.upscaleRestoreModel}_${AiConfig.upscaleMainModel}"

    private fun upscalerEngine(context: Context): PageUpscaler? {
        val files = chosenModelFiles(context)
        if (files.isEmpty() || files.any { it.length() == 0L }) return null
        val acceleration = if (acceleratorFailed) Acceleration.CPU else AiConfig.upscaleAcceleration
        val key = "models_${files.joinToString("|") { "${it.name}_${it.lastModified()}" }}_${acceleration}_${AiConfig.aiThreads}"
        return upscalerManager.getOrPut(key) {
            logcat(LogPriority.DEBUG) { "[Perf] Creating Anime4kStageUpscaler engine ($acceleration, threads=${AiConfig.aiThreads})" }
            Anime4kStageUpscaler(files, acceleration, AiConfig.aiThreads)
        }
    }

    private fun looksGrayscale(bitmap: Bitmap): Boolean {
        val side = 64
        val sample = Bitmap.createScaledBitmap(bitmap, side, side, true)
        try {
            val pixels = IntArray(side * side)
            sample.getPixels(pixels, 0, side, 0, 0, side, side)
            var coloredPixels = 0
            for (pixel in pixels) {
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                if (maxOf(r, g, b) - minOf(r, g, b) > 24) coloredPixels++
            }
            return coloredPixels < pixels.size / 20
        } finally {
            if (sample !== bitmap) sample.recycle()
        }
    }

    private fun hash(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private fun baseName(
        context: Context,
        imageHash: String,
        intensity: Float,
        upscale: Boolean,
        shrinkSide: Int,
        colorModel: File,
        sam: Boolean,
    ): String {
        val color = if (intensity < 0f) "n" else (intensity * 100).roundToInt().toString()
        val upscaleStamp = if (upscale) "${upscaleSignature()}_${AiConfig.upscaleMaxOutputPixels / 1_000_000}" else "n"
        val finalPassStamp = "_f${if (AiConfig.upscaleFinalPassEnabled) "1" else "0"}${if (AiConfig.upscaleFinalPassKeepSize) "k" else "g"}${AiConfig.upscaleFinalPassMaxPixels / 1_000_000}"
        return "${imageHash}_c${color}${if (sam) "s" else ""}_${colorModel.length()}_u${upscaleStamp}_s$shrinkSide${finalPassStamp}_t${(AiConfig.saturation * 100).roundToInt()}"
    }

    private fun cacheFile(context: Context, name: String): File = File(File(context.cacheDir, PAGE_CACHE_DIR), name)

    private fun readCachedPage(file: File): BufferedSource? {
        if (!file.isFile || file.length() == 0L) return null
        file.setLastModified(System.currentTimeMillis())
        return Buffer().write(file.readBytes())
    }

    private fun saveCachedPage(file: File, data: ByteArray) {
        if (AiConfig.isPost) return
        val dir = file.parentFile ?: return
        dir.mkdirs()
        val temp = File(dir, "${file.name}.${System.nanoTime()}.tmp")
        temp.writeBytes(data)
        temp.renameTo(file)
        trimCache(dir, AiConfig.cacheBytes)
    }

    private fun chromaFile(context: Context, imageHash: String, model: File): File =
        File(File(context.cacheDir, CHROMA_CACHE_DIR), imageHash + "_" + model.length() + (if (samActive(context)) "_sam" else ""))

    private fun loadChroma(file: File): MangaColorizer.Chroma? {
        if (!file.isFile) return null
        val bytes = file.readBytes()
        if (bytes.size != CHROMA_SIZE * 3 * Float.SIZE_BYTES) return null
        file.setLastModified(System.currentTimeMillis())
        val rgb = FloatArray(CHROMA_SIZE * 3)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(rgb)
        return MangaColorizer.Chroma(rgb)
    }

    private fun saveChroma(context: Context, file: File, chroma: MangaColorizer.Chroma) {
        val dir = file.parentFile ?: return
        dir.mkdirs()
        val buffer = ByteBuffer.allocate(CHROMA_SIZE * 3 * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        buffer.asFloatBuffer().put(chroma.rgb)
        val temp = File(dir, "${file.name}.${System.nanoTime()}.tmp")
        temp.writeBytes(buffer.array())
        temp.renameTo(file)
        trimCache(dir, MAX_CHROMA_CACHE_BYTES)
    }

    private fun trimCache(dir: File, limit: Long) {
        val files = dir.listFiles()
            ?.filter { !it.name.endsWith(".tmp") }
            ?.sortedBy { it.lastModified() }
            ?: return
        var total = files.sumOf { file -> file.length() }
        for (file in files) {
            if (total <= limit) break
            total -= file.length()
            file.delete()
        }
    }
}
