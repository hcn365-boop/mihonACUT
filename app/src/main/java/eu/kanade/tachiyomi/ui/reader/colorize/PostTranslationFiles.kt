package eu.kanade.tachiyomi.ui.reader.colorize

import android.content.Context
import android.net.Uri
import com.hippo.unifile.UniFile
import eu.kanade.translation.model.PageTranslation
import kotlinx.serialization.json.Json
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat

/**
 * The translation of a page as a json file in the "translations" folder of the chapter, named like the page
 * (001.jpg has 001.json). It is the same text the app keeps in its own storage, so it can live with the chapter
 * and survive a clear of the app data. Deleting the chapter deletes these files with it.
 */
object PostTranslationFiles {

    const val FOLDER = "translations"

    private val json = Json { ignoreUnknownKeys = true }

    /** @return the file that was written */
    fun write(folder: UniFile, name: String, page: PageTranslation): UniFile {
        folder.findFile(name)?.delete()
        val file = folder.createFile(name) ?: error("Cannot create $name")
        file.openOutputStream().use {
            it.write(json.encodeToString(PageTranslation.serializer(), page).toByteArray())
        }
        return file
    }

    /** @return the translation in the file, or null if it is gone or cannot be read */
    fun read(context: Context, uri: String): PageTranslation? = try {
        context.contentResolver.openInputStream(Uri.parse(uri))?.use {
            json.decodeFromString(PageTranslation.serializer(), it.readBytes().decodeToString())
        }
    } catch (e: Exception) {
        logcat(LogPriority.ERROR, e)
        null
    }
}
