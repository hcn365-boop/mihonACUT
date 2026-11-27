package eu.kanade.translation

import android.content.Context
import android.net.Uri
import com.hippo.unifile.UniFile
import eu.kanade.tachiyomi.ui.reader.colorize.AiConfig
import eu.kanade.tachiyomi.ui.reader.colorize.PageColorizer
import eu.kanade.tachiyomi.ui.reader.colorize.PostProcessStore
import eu.kanade.tachiyomi.ui.reader.colorize.PostRead
import java.io.File

/**
 * Deletes saved translations, by what the user has open:
 * - [deleteChapter]: the chapter open in the reader;
 * - [deleteManga]: the whole manga open in the reader;
 * - [deleteAll]: everything (the only choice in the general settings, where no manga is open).
 *
 * A translation kept next to a chapter (translations.json) is deleted with the saved text, otherwise it would
 * come back the next time the page is read. The original pages and the processed images are never touched.
 */
object TranslationCleaner {

    /**
     * The chapter open in the reader. The pages are found by the fingerprint of their bytes, so it works for a
     * chapter read online or downloaded.
     * @return how many saved texts were deleted
     */
    fun deleteChapter(context: Context): Int {
        val pages = MangaTranslationContext.currentChapterPages?.invoke().orEmpty()
        val hashes = pages.mapNotNull { bytes -> bytes()?.let { PageColorizer.contentHash(it) } }.toSet()
        if (hashes.isEmpty()) return 0

        // The file of batch processing holds every page of the chapter, so it goes too.
        val chapters = hashes.mapNotNull { PostProcessStore.chapterOfPage(context, it) }.toSet()
        deleteFiles(
            context,
            chapters.mapNotNull { id -> PostProcessStore.chapterTranslationUri(context, id)?.let { id to it } },
        )
        val deleted = TranslationStore.deleteForHashes(context, hashes)
        deleteCachedPages(context, hashes)
        reloadPages()
        return deleted
    }

    /**
     * The manga open in the reader. Texts saved before the app recorded the manga are removed only for the
     * chapter that is open (the app cannot know the others), or by [deleteAll].
     */
    fun deleteManga(context: Context): Int {
        val id = MangaTranslationContext.mangaId
        if (id < 0) return 0
        deleteFiles(context, PostProcessStore.translationFilesOfManga(context, id))
        // The pages are found before their rows are deleted, to clear the cached images made from them.
        val hashes = TranslationStore.hashesForManga(context, id)
        val deleted = TranslationStore.deleteForManga(context, id) + deleteChapter(context)
        deleteCachedPages(context, hashes)
        reloadPages()
        return deleted
    }

    fun deleteAll(context: Context) {
        deleteFiles(context, PostProcessStore.allTranslationFiles(context))
        TranslationStore.clear(context)
        deleteCachedPages(context, null)
        reloadPages()
    }

    /**
     * Deletes the cached images that have the translation drawn in them. Without this the reader would go on
     * showing the old pages from the cache, although the translation was deleted. The images made before the
     * text (colorized, upscaled) stay, so nothing has to be processed again.
     * @param hashes the pages to clean, or null for all of them
     */
    fun deleteCachedPages(context: Context, hashes: Set<String>?) {
        // Pages of the real-time pipeline: "<fingerprint>_..._T<key>" is a page with its text; the others are not.
        val withText = Regex("_T[0-9a-fA-F]+$")
        File(context.cacheDir, "ai_pages").listFiles()?.forEach { file ->
            if (withText.containsMatchIn(file.name) && (hashes == null || hashes.any { file.name.startsWith("${it}_") })) {
                file.delete()
            }
        }
        // Pages drawn when reading pages processed in advance.
        val drawn = File(context.cacheDir, "post_read")
        if (hashes == null) {
            drawn.deleteRecursively()
        } else {
            drawn.listFiles()?.forEach { file -> if (hashes.any { file.name.startsWith(it) }) file.delete() }
        }
    }

    private fun deleteFiles(context: Context, files: List<Pair<Long, String>>) {
        for ((chapterId, uri) in files) {
            runCatching { UniFile.fromUri(context, Uri.parse(uri))?.delete() }
            PostProcessStore.clearChapterTranslation(context, chapterId)
        }
    }

    /** The chapter files are read again when a page needs them, and the pages on screen are drawn again. */
    private fun reloadPages() {
        PostRead.forgetImports()
        ChapterTranslationFile.forget()
        AiConfig.requestReload()
    }
}
