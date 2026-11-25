package eu.kanade.tachiyomi.ui.reader.colorize

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import eu.kanade.translation.PageTranslationRenderer
import eu.kanade.translation.TranslationFont
import eu.kanade.translation.TranslationStore
import eu.kanade.translation.model.PageTranslation
import logcat.LogPriority
import okio.Buffer
import okio.BufferedSource
import tachiyomi.core.common.util.system.logcat
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * What the reader shows for a page that batch processing handled. The processed image has no text; the
 * translation was saved apart (under the fingerprint of the original page), so it is drawn now, which is
 * quick and needs no network. Changing the font or the padding therefore never needs a new translation.
 */
object PostRead {

    /** Chapter files already read in this run of the app. */
    private val imported = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()

    /** Call after the saved translations were deleted, so the chapter files are read again when needed. */
    fun forgetImports() = imported.clear()

    /** The name under which batch processing saves the translation of a page. */
    fun translationName(originalHash: String): String = "post_$originalHash"

    /**
     * @param original the bytes of the original page, used when batch processing left the image as it is
     * @return the page to show, or null when it must stay exactly as it is
     */
    fun compose(
        context: Context,
        originalHash: String,
        original: ByteArray,
        found: PostProcessStore.Found,
    ): BufferedSource? {
        val processed = (found as? PostProcessStore.Found.Processed)?.bytes
        val name = translationName(originalHash)
        var saved = TranslationStore.load(context, TranslationStore.Kind.TRANSLATED, name)
        if (saved == null) {
            // Kept next to the chapter: the file holds the whole chapter, so it is read ONCE, and every page
            // goes to the app's storage in one transaction. Reading it again for each page would be slow.
            val uri = PostProcessStore.translationUriForPage(context, originalHash)
            if (uri != null && imported.add(uri)) {
                val rows = PostTranslationFiles.readRows(context, uri)
                if (rows != null) {
                    TranslationStore.importRows(context, rows)
                    saved = TranslationStore.load(context, TranslationStore.Kind.TRANSLATED, name)
                }
            }
        }
        val translation = saved?.takeIf { it.blocks.isNotEmpty() }

        if (translation == null) return processed?.let { Buffer().write(it) }

        // Drawing means decoding a big image, drawing the text and encoding it again: far too slow to repeat
        // every time the page appears. The result is kept, and a new key is made whenever the translation, the
        // image, the font or the padding changes.
        val base = processed ?: original
        val font = AiConfig.translation?.font ?: TranslationFont.SYSTEM
        val key = buildString {
            append(originalHash)
            append('_').append(base.size)
            append('_').append(translation.blocks.size)
            append('_').append(translation.blocks.joinToString("|") { it.translation }.hashCode())
            append('_').append(font.name)
            append('_').append((AiConfig.translateCollisionPadding * 100).toInt())
        }
        readCached(context, key)?.let { return Buffer().write(it) }

        val drawn = drawOn(context, base, translation)
            ?: return processed?.let { Buffer().write(it) }
        saveCached(context, key, drawn)
        return Buffer().write(drawn)
    }

    private fun cacheDir(context: Context): File = File(context.cacheDir, "post_read")

    private fun readCached(context: Context, key: String): ByteArray? {
        val file = File(cacheDir(context), key)
        if (!file.isFile || file.length() == 0L) return null
        file.setLastModified(System.currentTimeMillis())
        return file.readBytes()
    }

    private fun saveCached(context: Context, key: String, bytes: ByteArray) {
        runCatching {
            val dir = cacheDir(context)
            dir.mkdirs()
            val temp = File(dir, "$key.${System.nanoTime()}.tmp")
            temp.writeBytes(bytes)
            temp.renameTo(File(dir, key))
            // The oldest pages go first once the folder is bigger than the cache size of the settings.
            val files = dir.listFiles()?.filter { !it.name.endsWith(".tmp") }?.sortedBy { it.lastModified() } ?: return
            var total = files.sumOf { it.length() }
            for (file in files) {
                if (total <= AiConfig.cacheBytes) break
                total -= file.length()
                file.delete()
            }
        }
    }

    /** Draws the translation on the image, or returns null if that cannot be done. */
    fun drawOn(context: Context, bytes: ByteArray, page: PageTranslation): ByteArray? {
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
        var rendered: Bitmap? = null
        return try {
            val font = AiConfig.translation?.font ?: TranslationFont.SYSTEM
            val result = PageTranslationRenderer.render(bitmap, page, font.typeface(context))
            if (result !== bitmap) rendered = result
            val format = if (bitmap.hasAlpha()) Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG
            ByteArrayOutputStream().also { result.compress(format, 95, it) }.toByteArray()
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e)
            null
        } finally {
            rendered?.recycle()
            bitmap.recycle()
        }
    }
}
