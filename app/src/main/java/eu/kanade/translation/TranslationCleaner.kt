package eu.kanade.translation

import android.content.Context
import android.net.Uri
import com.hippo.unifile.UniFile
import eu.kanade.tachiyomi.ui.reader.colorize.AiConfig
import eu.kanade.tachiyomi.ui.reader.colorize.PageColorizer
import eu.kanade.tachiyomi.ui.reader.colorize.PostProcessStore
import eu.kanade.tachiyomi.ui.reader.colorize.PostRead

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
        val deleted = TranslationStore.deleteForManga(context, id) + deleteChapter(context)
        reloadPages()
        return deleted
    }

    fun deleteAll(context: Context) {
        deleteFiles(context, PostProcessStore.allTranslationFiles(context))
        TranslationStore.clear(context)
        reloadPages()
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
        AiConfig.requestReload()
    }
}
