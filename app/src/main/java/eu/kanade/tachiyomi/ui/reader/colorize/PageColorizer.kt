package eu.kanade.tachiyomi.ui.reader.colorize

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import eu.kanade.translation.MangaTranslationContext
import eu.kanade.translation.PageTranslationRenderer
import eu.kanade.translation.PageTranslationService
import eu.kanade.translation.TranslationStore
import eu.kanade.translation.translator.TranslatorSettings
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
 * Processes a manga page for the reader: colorize, then upscale, then translate (text drawn into the
 * image). Settings come from [AiConfig].
 *
 * It never throws: if anything goes wrong (no model, huge image, already colored, out of memory...)
 * the original page is returned, so reading is never broken by this feature.
 *
 * Two images are cached: the "base" (colorized and upscaled, no text) and the "final" (base with the
 * translation drawn). Changing only the translation or the font reuses the base, and the translated
 * text itself is kept in [TranslationStore], so nothing is read or translated twice.
 */
object PageColorizer {

    private const val MODEL_DIR = "colorizer"
    private const val COLORIZER_FILE = "manga-colorize-fp16.onnx"
    private const val UPSCALER_FILE = "anime4k_acnet.onnx"
    private const val CHROMA_CACHE_DIR = "colorizer_cache"
    private const val PAGE_CACHE_DIR = "ai_pages"

    private const val MAX_CHROMA_CACHE_BYTES = 256L * 1024 * 1024
    private const val JPEG_QUALITY = 95

    private val mutex = Mutex()

    private val pageLocks = Array(16) { Mutex() }

    private fun pageLock(name: String): Mutex = pageLocks[(name.hashCode() and Int.MAX_VALUE) % pageLocks.size]

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

    // Manga Colorization V2 does not use Manga Light V6 SAM feature maps.
    private fun samActive(@Suppress("UNUSED_PARAMETER") context: Context): Boolean = false

    fun isModelInstalled(context: Context): Boolean = modelFile(context).length() > 0

    private fun acnetReady(context: Context): Boolean = true

    fun releaseEngines() {
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
            logcat(LogPriority.ERROR, e) { "PageColorizer: process failed, returning original source" }
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
            if (full) run(app, source, display = false, translate = translate) else warmChroma(app, source)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e) { "PageColorizer: prefetch failed" }
        }
    }

    private suspend fun run(
        context: Context,
        source: BufferedSource,
        display: Boolean,
        translate: Boolean = true,
    ): BufferedSource? {
        if (display && !AiConfig.isPost && AiConfig.usePostProcessed() && PostProcessStore.maybeHasResults) {
            val original = source.peek().readByteArray()
            val originalHash = hash(original)
            PostProcessStore.lookup(context, originalHash)?.let { found ->
                return PostRead.compose(context, originalHash, original, found)
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
            logcat(LogPriority.ERROR) {
                "PageColorizer: skipped (enabled=${AiConfig.colorizeEnabled}, intensity=$intensity, modelLen=${colorModel.length()}, file=${colorModel.absolutePath})"
            }
            return null
        }
        if (ImageUtil.isAnimatedAndSupported(source)) return null

        val bytes = source.peek().readByteArray()
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        val pixels = bounds.outWidth.toLong() * bounds.outHeight

        val doColor = wantColor && pixels <= AiConfig.colorizeMaxPixels
        if (wantColor && !doColor) {
            logcat(LogPriority.ERROR) {
                "PageColorizer: color skipped by pixels ($pixels > ${AiConfig.colorizeMaxPixels})"
            }
        }
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
        if (!doColor && !doUpscale && !doShrink && !wantFinal && translationConfig == null) return null

        val imageHash = hash(bytes)
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

        if (display && finalName != null) readCachedPage(cacheFile(context, finalName))?.let { return it }

        val base = if (doColor || doUpscale || doShrink || wantFinal) {
            baseImage(context, bytes, imageHash, baseName, doColor, doUpscale, shrinkSide, intensity, pixels, wantFinal)
        } else {
            null
        }
        if (translationConfig == null) return base?.let { Buffer().write(it) }

        if (!display) {
            var ocrBitmap: Bitmap? = null
            try {
                val s = translationConfig
                val targetBytes = base ?: bytes
                PageTranslationService.enqueue(context, effective!!, s.ocrMode, imageHash, s.limits) {
                    decode(targetBytes).also { ocrBitmap = it }
                }
            } finally {
                ocrBitmap?.recycle()
            }
            return base?.let { Buffer().write(it) }
        }

        return translateAndDraw(context, translationConfig, effective!!, imageHash, base ?: bytes, finalName!!)
            ?: base?.let { Buffer().write(it) }
    }

    private fun capSize(bitmap: Bitmap): Bitmap = PageResizer.fitTo(bitmap, AiConfig.upscaleMaxOutputPixels)

    private fun acnetFinalPass(context: Context, page: Bitmap): Bitmap {
        val pixels = page.width.toLong() * page.height
        if (pixels > AiConfig.upscaleFinalPassMaxPixels) return page
        if (pixels * (14 + 4 * 21) > AiConfig.memoryBudgetBytes) return page
        val acceleration = if (acceleratorFailed) Acceleration.CPU else AiConfig.upscaleAcceleration
        val key = "${acceleration}_${AiConfig.aiThreads}"

        val engine = acnetManager.getOrPut(key) { Anime4kUpscaler(context, acceleration, AiConfig.aiThreads) }

        val enlarged = engine.upscale(page)
        if (!AiConfig.upscaleFinalPassKeepSize) return capSize(enlarged)
        val restored = PageResizer.resizeTo(enlarged, page.width, page.height)
        if (restored !== enlarged) enlarged.recycle()
        return restored
    }

    private suspend fun baseImage(
        context: Context,
        bytes: ByteArray,
        imageHash: String,
        baseName: String,
        doColor: Boolean,
        doUpscale: Boolean,
        shrinkSide: Int,
        intensity: Float,
        pixels: Long,
        wantFinal: Boolean,
    ): ByteArray? {
        val file = cacheFile(context, baseName)
        if (file.isFile && file.length() > 0L) {
            file.setLastModified(System.currentTimeMillis())
            logcat(LogPriority.ERROR) { "PageColorizer: cache hit baseName=$baseName" }
            return file.readBytes()
        }

        val original = decode(bytes) ?: return null
        val working = if (shrinkSide > 0) PageResizer.shrink(original, shrinkSide) else original
        var result: Bitmap? = null
        try {
            var color = doColor
            val upscale = doUpscale && !original.hasAlpha()
            val finalPass = wantFinal && acnetReady(context) && !original.hasAlpha()
            if (color && !looksGrayscale(working)) {
                logcat(LogPriority.ERROR) { "PageColorizer: color skipped because looksGrayscale=false" }
                color = false
            }
            if (!color && !upscale && !finalPass && working === original) return null
            val estimated = estimateBytes(pixels, color, upscale, upscaleScale())
            if (estimated > AiConfig.memoryBudgetBytes) {
                logcat(LogPriority.ERROR) {
                    "PageColorizer: skipped by memoryBudget ($estimated > ${AiConfig.memoryBudgetBytes})"
                }
                return null
            }

            val processed = try {
                mutex.withLock {
                    var page = working
                    if (color) {
                        val colorModel = modelFile(context)
                        val chromaFile = chromaFile(context, imageHash, colorModel)
                        val engine = colorizerEngine(context, colorModel)
                        val fromCache = loadChroma(chromaFile)
                        val chroma = fromCache ?: engine.predict(working).also {
                            runCatching { saveChroma(context, chromaFile, it) }
                        }
                        logcat(LogPriority.ERROR) {
                            "PageColorizer: color chromaFromCache=${fromCache != null} intensity=$intensity sat=${AiConfig.saturation}"
                        }
                        page = engine.render(working, chroma, intensity)
                    }
                    if (upscale) {
                        val enlarged = (upscalerEngine(context) ?: error("No upscaler model"))
                            .process(page, AiConfig.upscaleMaxOutputPixels, PageResizer.MAX_SIDE)
                        if (page !== working) page.recycle()
                        page = enlarged
                    }
                    if (finalPass) {
                        val refined = acnetFinalPass(context, page)
                        if (refined !== page && page !== working) page.recycle()
                        page = refined
                    }

                    val saturation = AiConfig.saturation
                    if (color && saturation != 1f) {
                        val saturated = applySaturation(page, saturation)
                        if (saturated !== page && page !== working) page.recycle()
                        saturated
                    } else {
                        page
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                logcat(LogPriority.ERROR, e) { "PageColorizer: baseImage failed" }
                val accelerated = (color && AiConfig.colorizeAcceleration != Acceleration.CPU) ||
                    (upscale && AiConfig.upscaleAcceleration != Acceleration.CPU)
                if (accelerated && !acceleratorFailed) {
                    acceleratorFailed = true
                    mutex.withLock { releaseEngines() }
                }
                throw e
            }
            result = processed

            val encoded = encode(processed, original.hasAlpha())
            runCatching { saveCachedPage(file, encoded) }
            return encoded
        } finally {
            result?.recycle()
            if (working !== original) working.recycle()
            original.recycle()
        }
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

    private suspend fun translateAndDraw(
        context: Context,
        config: TranslationConfig,
        settings: TranslatorSettings,
        imageHash: String,
        baseBytes: ByteArray,
        finalName: String,
    ): BufferedSource? {
        var ocrBitmap: Bitmap? = null
        val page = try {
            PageTranslationService.translateNow(
                context,
                settings,
                config.ocrMode,
                imageHash,
                config.limits,
            ) { decode(baseBytes).also { ocrBitmap = it } }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e)
            null
        } finally {
            ocrBitmap?.recycle()
        } ?: return null

        val base = decode(baseBytes) ?: return null
        var drawn: Bitmap? = null
        try {
            val result = PageTranslationRenderer.render(base, page, config.font.typeface(context))
            if (result !== base) drawn = result
            val encoded = encode(result, base.hasAlpha())
            runCatching { saveCachedPage(cacheFile(context, finalName), encoded) }
            return Buffer().write(encoded)
        } finally {
            drawn?.recycle()
            base.recycle()
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
        val chromaFile = chromaFile(context, hash(bytes), model)
        if (chromaFile.isFile) return
        val original = decode(bytes) ?: return
        try {
            if (!looksGrayscale(original)) return
            mutex.withLock {
                if (!chromaFile.isFile) saveChroma(context, chromaFile, colorizerEngine(context, model).predict(original))
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
            MangaColorizer(model, acceleration, AiConfig.aiThreads, null)
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
        return upscalerManager.getOrPut(key) { Anime4kStageUpscaler(files, acceleration, AiConfig.aiThreads) }
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
        File(File(context.cacheDir, CHROMA_CACHE_DIR), imageHash + "_" + model.length() + "_v2")

    // Header: 4 bytes width + 4 bytes height (Big Endian), followed by planeSize bytes for A and planeSize bytes for B.
    private fun loadChroma(file: File): MangaColorizer.Chroma? {
        if (!file.isFile) return null
        val data = file.readBytes()
        if (data.size < 8) return null
        val header = ByteBuffer.wrap(data, 0, 8).order(ByteOrder.BIG_ENDIAN)
        val w = header.int
        val h = header.int
        if (w <= 0 || h <= 0 || w > 4096 || h > 4096) return null
        val plane = w.toLong() * h.toLong()
        if (plane > Int.MAX_VALUE / 2) return null
        val planeInt = plane.toInt()
        if (data.size != 8 + planeInt * 2) return null
        file.setLastModified(System.currentTimeMillis())
        return MangaColorizer.Chroma(
            width = w,
            height = h,
            a = FloatArray(planeInt) { data[8 + it].toFloat() },
            b = FloatArray(planeInt) { data[8 + planeInt + it].toFloat() },
        )
    }

    private fun saveChroma(context: Context, file: File, chroma: MangaColorizer.Chroma) {
        val dir = file.parentFile ?: return
        dir.mkdirs()
        val plane = chroma.planeSize
        val data = ByteArray(8 + plane * 2)
        val header = ByteBuffer.wrap(data, 0, 8).order(ByteOrder.BIG_ENDIAN)
        header.putInt(chroma.width)
        header.putInt(chroma.height)
        for (i in 0 until plane) {
            data[8 + i] = chroma.a[i].coerceIn(-127f, 127f).roundToInt().toByte()
            data[8 + plane + i] = chroma.b[i].coerceIn(-127f, 127f).roundToInt().toByte()
        }
        val temp = File(dir, "${file.name}.${System.nanoTime()}.tmp")
        temp.writeBytes(data)
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
