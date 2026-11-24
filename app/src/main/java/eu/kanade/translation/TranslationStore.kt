package eu.kanade.translation

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import eu.kanade.tachiyomi.ui.reader.colorize.AiConfig
import eu.kanade.translation.model.PageTranslation
import kotlinx.serialization.json.Json
import java.security.MessageDigest

/**
 * Keeps what was read and translated from each page, in one SQLite database in the app's private
 * storage (not in the cache, so the system does not wipe it). A page that is in the store is never
 * sent to OCR or to a translator again.
 *
 * One database means a lookup is a single indexed query, with no folder of thousands of tiny files.
 * Two kinds are kept apart, so changing the translation engine or language redoes only the translation
 * and not the OCR.
 */
object TranslationStore {

    enum class Kind(val id: Int) {
        OCR(0),
        TRANSLATED(1),

        /** The language the OCR of a page was run with, as a plain name. */
        LANGUAGE(2),
    }

    private const val DATABASE = "translations.db"
    private const val TABLE = "pages"

    private val json = Json { ignoreUnknownKeys = true }

    @Volatile
    private var helper: Helper? = null

    private class Helper(context: Context) : SQLiteOpenHelper(context, DATABASE, null, 2) {
        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL(
                "CREATE TABLE $TABLE (name TEXT NOT NULL, kind INTEGER NOT NULL, data TEXT NOT NULL, " +
                    "updated INTEGER NOT NULL, manga INTEGER NOT NULL DEFAULT -1, PRIMARY KEY (name, kind))",
            )
            db.execSQL("CREATE INDEX IF NOT EXISTS idx_${TABLE}_manga ON $TABLE(manga)")
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            if (oldVersion < 2) {
                db.execSQL("ALTER TABLE $TABLE ADD COLUMN manga INTEGER NOT NULL DEFAULT -1")
            }
            db.execSQL("CREATE INDEX IF NOT EXISTS idx_${TABLE}_manga ON $TABLE(manga)")
        }
    }

    private fun database(context: Context): SQLiteDatabase {
        val current = helper ?: synchronized(this) {
            helper ?: Helper(context.applicationContext).also { helper = it }
        }
        return current.writableDatabase
    }

    /** A short stable name for a set of settings, used inside names. */
    fun key(vararg parts: String): String =
        MessageDigest.getInstance("SHA-256").digest(parts.joinToString("|").toByteArray())
            .joinToString("") { "%02x".format(it) }
            .take(16)

    fun load(context: Context, kind: Kind, name: String): PageTranslation? {
        val db = database(context)
        val data = db.rawQuery(
            "SELECT data FROM $TABLE WHERE name = ? AND kind = ?",
            arrayOf(name, kind.id.toString()),
        ).use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null } ?: return null
        return try {
            json.decodeFromString(PageTranslation.serializer(), data)
        } catch (_: Exception) {
            // A damaged row is removed so the page is processed again.
            db.delete(TABLE, "name = ? AND kind = ?", arrayOf(name, kind.id.toString()))
            null
        }
    }

    fun save(context: Context, kind: Kind, name: String, page: PageTranslation, manga: Long = AiConfig.mangaId) {
        val values = ContentValues().apply {
            put("name", name)
            put("manga", manga)
            put("kind", kind.id)
            put("data", json.encodeToString(PageTranslation.serializer(), page))
            put("updated", System.currentTimeMillis())
        }
        database(context).insertWithOnConflict(TABLE, null, values, SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun loadText(context: Context, kind: Kind, name: String): String? =
        database(context).rawQuery(
            "SELECT data FROM $TABLE WHERE name = ? AND kind = ?",
            arrayOf(name, kind.id.toString()),
        ).use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }

    fun saveText(context: Context, kind: Kind, name: String, value: String, manga: Long = AiConfig.mangaId) {
        val values = ContentValues().apply {
            put("name", name)
            put("manga", manga)
            put("kind", kind.id)
            put("data", value)
            put("updated", System.currentTimeMillis())
        }
        database(context).insertWithOnConflict(TABLE, null, values, SQLiteDatabase.CONFLICT_REPLACE)
    }

    /** Saves many pages in one transaction: much faster than one save after another. */
    fun saveAll(context: Context, kind: Kind, pages: Map<String, PageTranslation>, manga: Long = AiConfig.mangaId) {
        if (pages.isEmpty()) return
        val db = database(context)
        db.beginTransaction()
        try {
            val now = System.currentTimeMillis()
            val values = ContentValues()
            for ((name, page) in pages) {
                values.clear()
                values.put("name", name)
                values.put("kind", kind.id)
                values.put("manga", manga)
                values.put("data", json.encodeToString(PageTranslation.serializer(), page))
                values.put("updated", now)
                db.insertWithOnConflict(TABLE, null, values, SQLiteDatabase.CONFLICT_REPLACE)
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    /**
     * Deletes everything saved for the pages with these fingerprints: their text, their language, and the
     * copies made by batch processing. Names start with the fingerprint of the page.
     * @return how many rows were deleted
     */
    fun deleteForHashes(context: Context, hashes: Collection<String>): Int {
        if (hashes.isEmpty()) return 0
        val db = database(context)
        var deleted = 0
        db.beginTransaction()
        try {
            for (hash in hashes) {
                deleted += db.delete(
                    TABLE,
                    "name LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\'",
                    arrayOf("${escapeLike(hash)}%", "post_${escapeLike(hash)}%"),
                )
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
        return deleted
    }

    private fun escapeLike(value: String): String =
        value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    /**
     * Deletes everything saved while reading or processing one manga. Rows saved before this existed have no
     * manga and are only removed by [clear].
     * @return how many rows were deleted
     */
    fun deleteForManga(context: Context, mangaId: Long): Int =
        database(context).delete(TABLE, "manga = ?", arrayOf(mangaId.toString()))

    /** Deletes everything of one kind, or of both kinds when [kind] is null. */
    fun clear(context: Context, kind: Kind? = null) {
        val db = database(context)
        if (kind == null) db.delete(TABLE, null, null) else db.delete(TABLE, "kind = ?", arrayOf(kind.id.toString()))
    }

    fun sizeBytes(context: Context): Long = context.getDatabasePath(DATABASE).length()

    @Synchronized
    fun close() {
        helper?.close()
        helper = null
    }
}
