package eu.kanade.presentation.reader.settings

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import eu.kanade.tachiyomi.ui.reader.colorize.AiPrefs
import eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences
import tachiyomi.presentation.core.components.CheckboxItem

/**
 * Settings of the batch processing: what the button next to a downloaded chapter does. They are kept
 * apart from the real-time settings (colorize, upscale and translate while reading), so the two can
 * differ, for example a heavier upscaler for chapters processed in advance.
 */
@Composable
internal fun BatchSettings(base: ReaderPreferences) {
    val prefs = remember { AiPrefs(base, null, "_post") }
    var open by remember { mutableIntStateOf(-1) }

    CheckboxItem(label = "Process chapters automatically after they are downloaded", pref = base.postProcessAuto)
    CheckboxItem(label = "Use the processed pages when reading", pref = base.postUseProcessed)
    Text(
        text = "Processed pages are saved in a folder named \"processed\" inside the chapter's download folder, " +
            "next to the original pages, which are never changed.",
        style = MaterialTheme.typography.bodySmall,
        modifier = Modifier.padding(horizontal = 16.dp, vertical = 2.dp),
    )
    TextButton(
        onClick = { prefs.copyGeneralSettings() },
        modifier = Modifier.padding(horizontal = 8.dp),
    ) {
        Text("Copy the real-time settings here")
    }

    Section("Colorize", open == 0, { open = if (open == 0) -1 else 0 }) { ColorizeSettings(prefs) }
    Section("Upscale", open == 1, { open = if (open == 1) -1 else 1 }) { UpscaleSettings(prefs) }
    Section("Translate", open == 2, { open = if (open == 2) -1 else 2 }) {
        TranslationSection(prefs, inReader = false)
    }
}

/** A title that opens and closes its settings. */
@Composable
private fun Section(title: String, expanded: Boolean, onToggle: () -> Unit, content: @Composable () -> Unit) {
    TextButton(
        onClick = onToggle,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp),
    ) {
        Text(
            text = (if (expanded) "▾  " else "▸  ") + title,
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.fillMaxWidth(),
        )
    }
    if (expanded) {
        Column { content() }
    }
}
