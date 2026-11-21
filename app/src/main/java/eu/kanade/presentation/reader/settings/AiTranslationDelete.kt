package eu.kanade.presentation.reader.settings

import android.widget.Toast
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import eu.kanade.translation.MangaTranslationContext
import eu.kanade.translation.TranslationCleaner
import eu.kanade.translation.TranslationStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private enum class DeleteAction(val button: String, val question: String) {
    CHAPTER(
        "Delete the translations of this chapter",
        "The saved translations of this chapter are deleted. The pages are not changed, and the chapter is " +
            "translated again the next time it is read.",
    ),
    MANGA(
        "Delete the translations of this manga",
        "The saved translations of this whole manga are deleted. The pages are not changed.",
    ),
    ALL(
        "Delete all saved translations",
        "Every saved translation of every manga is deleted, also the translation files kept next to the " +
            "chapters. The pages are not changed.",
    ),
}

/**
 * The delete buttons follow where the user is: inside a manga (the reader) there are two, for the chapter and
 * for the whole manga; in the general settings there is only one, for everything.
 */
@Composable
internal fun TranslationDeleteButtons(inReader: Boolean) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var size by remember { mutableStateOf(TranslationStore.sizeBytes(context)) }
    var pending by remember { mutableStateOf<DeleteAction?>(null) }
    val inManga = inReader && MangaTranslationContext.mangaId >= 0

    if (inManga) {
        TextButton(onClick = { pending = DeleteAction.CHAPTER }, modifier = Modifier.padding(horizontal = 8.dp)) {
            Text(DeleteAction.CHAPTER.button)
        }
        TextButton(onClick = { pending = DeleteAction.MANGA }, modifier = Modifier.padding(horizontal = 8.dp)) {
            Text(DeleteAction.MANGA.button)
        }
    } else {
        TextButton(onClick = { pending = DeleteAction.ALL }, modifier = Modifier.padding(horizontal = 8.dp)) {
            Text("${DeleteAction.ALL.button} (${size / 1024} KB)")
        }
    }

    pending?.let { action ->
        AlertDialog(
            onDismissRequest = { pending = null },
            title = { Text(action.button + "?") },
            text = { Text(action.question) },
            confirmButton = {
                TextButton(
                    onClick = {
                        pending = null
                        scope.launch {
                            // Files on shared storage can be slow to delete, so this is not done on the main thread.
                            val deleted = withContext(Dispatchers.IO) {
                                when (action) {
                                    DeleteAction.CHAPTER -> TranslationCleaner.deleteChapter(context)
                                    DeleteAction.MANGA -> TranslationCleaner.deleteManga(context)
                                    DeleteAction.ALL -> {
                                        TranslationCleaner.deleteAll(context)
                                        -1
                                    }
                                }
                            }
                            size = TranslationStore.sizeBytes(context)
                            val message = if (deleted >= 0) "Deleted $deleted saved texts" else "Deleted"
                            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                        }
                    },
                ) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { pending = null }) { Text("Cancel") } },
        )
    }
}
