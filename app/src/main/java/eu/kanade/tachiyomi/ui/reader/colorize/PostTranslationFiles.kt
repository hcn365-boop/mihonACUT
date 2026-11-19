package eu.kanade.tachiyomi.ui.reader.colorize

import android.content.Context
import android.net.Uri
import com.hippo.unifile.UniFile
import eu.kanade.translation.model.PageTranslation
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat

/**
 * The translations of a whole chapter in ONE json file, "translations.json", in the chapter folder (like the
 * ComicInfo.xml file the downloader writes there). It holds the same text the app keeps in its own storage, so
 * it lives with the chapter, survives a clear of the app data, and is deleted with the chapter.
 *
 * One file, written once when the chapter is done and read once when it is first opened, because every file
 * operation on shared storage is slow: a file for each page would cost that many times over.
 *
 * The pages are keyed by the fingerprint of the bytes the reader reads for them, not by file names.
 */
object PostTranslationFiles {

    const val FILE = "translations.json"

    private val json = Json { ignoreUnknownKeys = true }
    private val serializer = MapSerializer(String.serializer(), PageTranslation.serializer())

    /** @return the file that was written */
    fun writeChapter(chapterDir: UniFile, pages: Map<String, PageTranslation>): UniFile {
        chapterDir.findFile(FILE)?.delete()
        val file = chapterDir.createFile(FILE) ?: error("Cannot create $FILE")
        file.openOutputStream().use { it.write(json.encodeToString(serializer, pages).toByteArray()) }
        return file
    }

    /** @return the translations of the chapter, or null if the file is gone or cannot be read */
    fun readChapter(context: Context, uri: String): Map<String, PageTranslation>? = try {
        context.contentResolver.openInputStream(Uri.parse(uri))?.use {
            json.decodeFromString(serializer, it.readBytes().decodeToString())
        }
    } catch (e: Exception) {
        logcat(LogPriority.ERROR, e)
        null
    }
}
