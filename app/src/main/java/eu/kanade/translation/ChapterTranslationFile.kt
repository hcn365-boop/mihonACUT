package eu.kanade.translation

import android.content.Context
import com.hippo.unifile.UniFile
import eu.kanade.tachiyomi.ui.reader.colorize.PostProcessStore
import eu.kanade.tachiyomi.ui.reader.colorize.PostTranslationFiles
import java.util.concurrent.ConcurrentHashMap

/**
 * Keeps the translations made while reading in the chapter's folder (when the user chose to keep the
 * translation next to the chapter), so they survive a clear of the app data and travel with the chapter.
 *
 * - When a chapter is opened, its file is loaded into the app's storage ONCE; the rows carry the translation
 *   settings in their names, so only the ones made with the settings in use are found, and the translation
 *   goes on from where it stopped.
 * - While reading, the fingerprint of every page the translation looked at is remembered, with no cost.
 * - When the chapter is left, the rows of those pages are saved in the file ONCE, and only if something is new.
 */
object ChapterTranslationFile {

    /** Files read in this run of the app, with the rows they had. */
    private val loaded = ConcurrentHashMap<String, Set<String>>()

    /** Fingerprints of the pages translated since the last save. */
    private val seen: MutableSet<String> = ConcurrentHashMap.newKeySet()

    private const val MAX_SEEN = 5000

    /** Called for every page the translation looks at. */
    fun noteSeen(hash: String) {
        if (seen.size < MAX_SEEN) seen.add(hash)
    }

    /** Loads the file of a chapter folder, if it has one and it was not loaded yet. */
    fun load(context: Context, chapterDir: UniFile) {
        val file = chapterDir.findFile(PostTranslationFiles.FILE)?.takeIf { it.isFile } ?: return
        val uri = file.uri.toString()
        if (loaded.containsKey(uri)) return
        val rows = PostTranslationFiles.readRows(context, uri).orEmpty()
        TranslationStore.importRows(context, rows)
        loaded[uri] = rows.keys
    }

    /** Saves the rows of the pages seen since the last save into the chapter's file, if there is something new. */
    fun saveSeen(context: Context, chapterDir: UniFile, chapterId: Long, mangaId: Long) {
        val hashes = seen.toList()
        seen.clear()
        if (hashes.isEmpty()) return
        val rows = TranslationStore.exportForHashes(context, hashes)
        if (rows.isEmpty()) return

        // What the file already holds stays: it may have pages that were not seen this time.
        val existing = chapterDir.findFile(PostTranslationFiles.FILE)?.takeIf { it.isFile }
        val merged = LinkedHashMap<String, String>()
        existing?.let { PostTranslationFiles.readRows(context, it.uri.toString()) }?.let { merged.putAll(it) }
        merged.putAll(rows)
        if (existing != null && loaded[existing.uri.toString()] == merged.keys) return

        val file = PostTranslationFiles.writeRows(chapterDir, merged)
        // Remembered with the manga, so deleting the translations of a manga finds this file.
        PostProcessStore.registerChapter(context, chapterId, mangaId)
        PostProcessStore.setChapterTranslationUri(context, chapterId, file.uri.toString())
        loaded[file.uri.toString()] = merged.keys
    }

    /** The files are read again when needed, for example after the saved translations were deleted. */
    fun forget() {
        loaded.clear()
    }
}
