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

/**
 * What the reader shows for a page that batch processing handled. The processed image has no text; the
 * translation was saved apart (under the fingerprint of the original page), so it is drawn now, which is
 * quick and needs no network. Changing the font or the padding therefore never needs a new translation.
 */
object PostRead {

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
            // Kept next to the chapter: read the file, and keep a copy in the app for the next time.
            PostProcessStore.translationUri(context, originalHash)?.let { uri ->
                saved = PostTranslationFiles.read(context, uri)?.also {
                    TranslationStore.save(context, TranslationStore.Kind.TRANSLATED, name, it)
                }
            }
        }
        val translation = saved?.takeIf { it.blocks.isNotEmpty() }

        if (translation == null) return processed?.let { Buffer().write(it) }
        val drawn = drawOn(context, processed ?: original, translation)
            ?: return processed?.let { Buffer().write(it) }
        return Buffer().write(drawn)
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
