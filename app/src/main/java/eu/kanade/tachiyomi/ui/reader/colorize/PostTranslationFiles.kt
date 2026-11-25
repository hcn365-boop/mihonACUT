package eu.kanade.tachiyomi.ui.reader.colorize

import android.content.Context
import android.net.Uri
import com.hippo.unifile.UniFile
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat

/**
 * The translations of a whole chapter in ONE json file, "translations.json", in the chapter folder (like the
 * ComicInfo.xml file the downloader writes there). It is a copy of the rows the app keeps for the pages of the
 * chapter, so it lives with the chapter, survives a clear of the app data, and is deleted with the chapter.
 * A row carries the translation settings in its name, so when the file is loaded only the rows made with the
 * settings in use are found again.
 *
 * One file, written when a chapter is done or left and read once when it is opened: every file operation on
 * shared storage is slow, so a file for each page would cost that many times over.
 */
object PostTranslationFiles {

    const val FILE = "translations.json"

    private val json = Json { ignoreUnknownKeys = true }
    private val serializer = MapSerializer(String.serializer(), String.serializer())

    /** @return the file that was written */
    fun writeRows(chapterDir: UniFile, rows: Map<String, String>): UniFile {
        chapterDir.findFile(FILE)?.delete()
        val file = chapterDir.createFile(FILE) ?: error("Cannot create $FILE")
        file.openOutputStream().use { it.write(json.encodeToString(serializer, rows).toByteArray()) }
        return file
    }

    /** @return the rows in the file, or null if it is gone or cannot be read */
    fun readRows(context: Context, uri: String): Map<String, String>? = try {
        context.contentResolver.openInputStream(Uri.parse(uri))?.use {
            json.decodeFromString(serializer, it.readBytes().decodeToString())
        }
    } catch (e: Exception) {
        logcat(LogPriority.ERROR, e)
        null
    }
}
