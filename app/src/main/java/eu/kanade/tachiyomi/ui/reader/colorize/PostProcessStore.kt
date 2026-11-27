package eu.kanade.tachiyomi.ui.reader.colorize

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.net.Uri

/**
 * What the batch processing made. The processed pages are files in a "processed" folder inside the
 * downloaded chapter's folder; this small database only remembers, for the fingerprint of each original
 * page, where its processed file is (or that the page was left as it is), and which chapters are done.
 * If a file is gone (the chapter was deleted), its row is removed when the reader looks for it.
 */
object PostProcessStore {

    /** What the reader finds for an original page. */
    sealed interface Found {
        /** The page processed in advance. */
        class Processed(val bytes: ByteArray) : Found

        /** Batch processing looked at this page and left it as it is, so the reader must not process it either. */
        data object Keep : Found
    }

    /** The chapter is being processed. */
    const val STATE_RUNNING = 0
    const val STATE_PROCESSED = 1

    /** The translation of a chapter (a state kept apart from the one of its images). */
    const val TRANSLATION_DONE = 1
    const val TRANSLATION_ERROR = 2
    const val STATE_ERROR = 2

    /** Processed with the original pages replaced. */
    const val STATE_FINAL = 3

    @Volatile
    private var hasAny = false

    /** True when some page was processed in advance. Up to date after [refresh] ran once. */
    val maybeHasResults: Boolean
        get() = hasAny

    private var helper: Helper? = null

    private fun db(context: Context): SQLiteDatabase = synchronized(this) {
        helper ?: Helper(context.applicationContext).also { helper = it }
    }.writableDatabase

    /** Reads whether there are results. Call when the app opens a manga or the reader. */
    fun refresh(context: Context) {
        hasAny = db(context).rawQuery("SELECT 1 FROM pages LIMIT 1", null).use { it.moveToFirst() }
    }

    fun lookup(context: Context, hash: String): Found? {
        val uri = db(context).rawQuery("SELECT uri FROM pages WHERE hash = ?", arrayOf(hash))
            .use { if (it.moveToFirst()) it.getString(0) else null } ?: return null
        if (uri.isEmpty()) return Found.Keep
        val bytes = try {
            context.contentResolver.openInputStream(Uri.parse(uri))?.use { it.readBytes() }
        } catch (e: Exception) {
            null
        }
        if (bytes == null || bytes.isEmpty()) {
            removePage(context, hash)
            return null
        }
        return Found.Processed(bytes)
    }

    /** [uri] is the processed file, or empty when the page stays as it is. */
    fun savePage(context: Context, chapterId: Long, hash: String, uri: String) {
        val values = ContentValues().apply {
            put("hash", hash)
            put("chapter", chapterId)
            put("uri", uri)
        }
        db(context).insertWithOnConflict("pages", null, values, SQLiteDatabase.CONFLICT_REPLACE)
        hasAny = true
    }

    private fun removePage(context: Context, hash: String) {
        db(context).delete("pages", "hash = ?", arrayOf(hash))
        refresh(context)
    }

    /** Where the processed files of a chapter are (pages left as they are have no file). */
    fun chapterUris(context: Context, chapterId: Long): List<String> =
        db(context).rawQuery("SELECT uri FROM pages WHERE chapter = ? AND uri != ''", arrayOf(chapterId.toString()))
            .use { cursor -> buildList { while (cursor.moveToNext()) add(cursor.getString(0)) } }

    /** The translation file of a chapter (json), when the translation is kept next to the chapter. */
    fun chapterTranslationUri(context: Context, chapterId: Long): String? =
        db(context).rawQuery("SELECT translation FROM chapters WHERE chapter = ?", arrayOf(chapterId.toString()))
            .use { if (it.moveToFirst()) it.getString(0)?.takeIf { uri -> uri.isNotEmpty() } else null }

    fun setChapterTranslationUri(context: Context, chapterId: Long, uri: String) {
        val values = ContentValues().apply { put("translation", uri) }
        val updated = db(context).update("chapters", values, "chapter = ?", arrayOf(chapterId.toString()))
        if (updated == 0) {
            values.put("chapter", chapterId)
            values.put("state", STATE_RUNNING)
            values.put("updated", System.currentTimeMillis())
            db(context).insert("chapters", null, values)
        }
    }

    /** The translation file of the chapter a page belongs to. */
    fun translationUriForPage(context: Context, hash: String): String? =
        db(context).rawQuery(
            "SELECT c.translation FROM pages p JOIN chapters c ON c.chapter = p.chapter WHERE p.hash = ?",
            arrayOf(hash),
        ).use { if (it.moveToFirst()) it.getString(0)?.takeIf { uri -> uri.isNotEmpty() } else null }

    /** Remembers which manga a chapter belongs to, so the files of a whole manga can be found. */
    fun registerChapter(context: Context, chapterId: Long, mangaId: Long) {
        val values = ContentValues().apply { put("manga", mangaId) }
        val updated = db(context).update("chapters", values, "chapter = ?", arrayOf(chapterId.toString()))
        if (updated == 0) {
            values.put("chapter", chapterId)
            values.put("state", STATE_RUNNING)
            values.put("updated", System.currentTimeMillis())
            db(context).insert("chapters", null, values)
        }
    }

    /** The chapter (and its translation file) of every chapter of a manga that keeps its translation in a file. */
    fun translationFilesOfManga(context: Context, mangaId: Long): List<Pair<Long, String>> =
        db(context).rawQuery(
            "SELECT chapter, translation FROM chapters WHERE manga = ? AND translation IS NOT NULL AND translation != ''",
            arrayOf(mangaId.toString()),
        ).use { cursor -> buildList { while (cursor.moveToNext()) add(cursor.getLong(0) to cursor.getString(1)) } }

    fun allTranslationFiles(context: Context): List<Pair<Long, String>> =
        db(context).rawQuery(
            "SELECT chapter, translation FROM chapters WHERE translation IS NOT NULL AND translation != ''",
            null,
        ).use { cursor -> buildList { while (cursor.moveToNext()) add(cursor.getLong(0) to cursor.getString(1)) } }

    /** Forgets the translation file of a chapter (the caller deletes the file). */
    fun clearChapterTranslation(context: Context, chapterId: Long) {
        val values = ContentValues().apply { putNull("translation") }
        db(context).update("chapters", values, "chapter = ?", arrayOf(chapterId.toString()))
    }

    /** The chapter a page was processed with, from the fingerprint of the page. */
    fun chapterOfPage(context: Context, hash: String): Long? =
        db(context).rawQuery("SELECT chapter FROM pages WHERE hash = ?", arrayOf(hash))
            .use { if (it.moveToFirst()) it.getLong(0) else null }

    fun setTranslationState(context: Context, chapterId: Long, state: Int) {
        val values = ContentValues().apply { put("translation_state", state) }
        val updated = db(context).update("chapters", values, "chapter = ?", arrayOf(chapterId.toString()))
        if (updated == 0) {
            values.put("chapter", chapterId)
            values.put("state", STATE_RUNNING)
            values.put("updated", System.currentTimeMillis())
            db(context).insert("chapters", null, values)
        }
    }

    fun chaptersWithTranslationState(context: Context, state: Int): Set<Long> =
        db(context).rawQuery("SELECT chapter FROM chapters WHERE translation_state = ?", arrayOf(state.toString()))
            .use { cursor -> buildSet { while (cursor.moveToNext()) add(cursor.getLong(0)) } }

    /** True when batch processing already knows this page (processed, or marked "leave as it is"). */
    fun hasPage(context: Context, hash: String): Boolean =
        db(context).rawQuery("SELECT 1 FROM pages WHERE hash = ?", arrayOf(hash)).use { it.moveToFirst() }

    /** The fingerprints of the pages of a chapter that batch processing knows. */
    fun pageHashesOfChapter(context: Context, chapterId: Long): List<String> =
        db(context).rawQuery("SELECT hash FROM pages WHERE chapter = ?", arrayOf(chapterId.toString()))
            .use { cursor -> buildList { while (cursor.moveToNext()) add(cursor.getString(0)) } }

    fun chaptersWithState(context: Context, state: Int): Set<Long> =
        db(context).rawQuery("SELECT chapter FROM chapters WHERE state = ?", arrayOf(state.toString()))
            .use { cursor -> buildSet { while (cursor.moveToNext()) add(cursor.getLong(0)) } }

    fun setChapter(context: Context, chapterId: Long, state: Int) {
        val values = ContentValues().apply {
            put("state", state)
            put("updated", System.currentTimeMillis())
        }
        // Update, not replace: the row may already hold the chapter's translation file.
        val updated = db(context).update("chapters", values, "chapter = ?", arrayOf(chapterId.toString()))
        if (updated == 0) {
            values.put("chapter", chapterId)
            db(context).insert("chapters", null, values)
        }
    }

    fun processedChapters(context: Context): Set<Long> =
        db(context).rawQuery("SELECT chapter FROM chapters WHERE state = ?", arrayOf(STATE_PROCESSED.toString()))
            .use { cursor -> buildSet { while (cursor.moveToNext()) add(cursor.getLong(0)) } }

    /** Forgets everything about a chapter. The caller deletes the files. */
    fun deleteChapter(context: Context, chapterId: Long) {
        db(context).delete("pages", "chapter = ?", arrayOf(chapterId.toString()))
        db(context).delete("chapters", "chapter = ?", arrayOf(chapterId.toString()))
        refresh(context)
    }

    private class Helper(context: Context) : SQLiteOpenHelper(context, "post_process.db", null, 5) {
        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL("CREATE TABLE pages (hash TEXT PRIMARY KEY, chapter INTEGER NOT NULL, uri TEXT NOT NULL)")
            db.execSQL("CREATE INDEX pages_chapter ON pages (chapter)")
            db.execSQL(
                "CREATE TABLE chapters (chapter INTEGER PRIMARY KEY, state INTEGER NOT NULL, updated INTEGER NOT NULL, translation TEXT, manga INTEGER NOT NULL DEFAULT -1, translation_state INTEGER NOT NULL DEFAULT 0)",
            )
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            if (oldVersion < 3) db.execSQL("ALTER TABLE chapters ADD COLUMN translation TEXT")
            if (oldVersion < 4) db.execSQL("ALTER TABLE chapters ADD COLUMN manga INTEGER NOT NULL DEFAULT -1")
            if (oldVersion < 5) db.execSQL("ALTER TABLE chapters ADD COLUMN translation_state INTEGER NOT NULL DEFAULT 0")
        }
    }
}
