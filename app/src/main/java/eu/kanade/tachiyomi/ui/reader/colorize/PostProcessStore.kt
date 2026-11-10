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

    const val STATE_PROCESSED = 1
    const val STATE_ERROR = 2

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

    fun setChapter(context: Context, chapterId: Long, state: Int) {
        val values = ContentValues().apply {
            put("chapter", chapterId)
            put("state", state)
            put("updated", System.currentTimeMillis())
        }
        db(context).insertWithOnConflict("chapters", null, values, SQLiteDatabase.CONFLICT_REPLACE)
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

    private class Helper(context: Context) : SQLiteOpenHelper(context, "post_process.db", null, 1) {
        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL("CREATE TABLE pages (hash TEXT PRIMARY KEY, chapter INTEGER NOT NULL, uri TEXT NOT NULL)")
            db.execSQL("CREATE INDEX pages_chapter ON pages (chapter)")
            db.execSQL(
                "CREATE TABLE chapters (chapter INTEGER PRIMARY KEY, state INTEGER NOT NULL, updated INTEGER NOT NULL)",
            )
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit
    }
}
