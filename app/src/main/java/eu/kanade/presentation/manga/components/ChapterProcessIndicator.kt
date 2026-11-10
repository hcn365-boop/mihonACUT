package eu.kanade.presentation.manga.components

import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.hapticfeedback.HapticFeedback
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import eu.kanade.presentation.components.DropdownMenu
import eu.kanade.tachiyomi.ui.reader.colorize.ChapterProcessState
import mihon.icons.materialsymbols.MaterialSymbols
import mihon.icons.materialsymbols.rounded.Info
import mihon.icons.materialsymbols.rounded.Palette
import tachiyomi.presentation.core.components.material.IconButtonTokens
import tachiyomi.presentation.core.util.secondaryItemAlpha

enum class ChapterProcessAction {
    START,
    CANCEL,
    DELETE,
}

/**
 * The processing button of a chapter, shown next to its download button. Press to process the downloaded
 * pages with the post-download settings; while it works a ring shows the progress and a menu cancels it;
 * a processed chapter has a menu to delete the processed pages.
 */
@Composable
fun ChapterProcessIndicator(
    enabled: Boolean,
    stateProvider: () -> ChapterProcessState,
    onClick: (ChapterProcessAction) -> Unit,
    modifier: Modifier = Modifier,
) {
    when (val state = stateProvider()) {
        ChapterProcessState.NotProcessed -> IdleIndicator(enabled, modifier, onClick)
        ChapterProcessState.Queued -> WorkingIndicator(enabled, null, modifier, onClick)
        is ChapterProcessState.Processing -> WorkingIndicator(enabled, state.progress, modifier, onClick)
        ChapterProcessState.Processed -> DoneIndicator(enabled, modifier, onClick)
        ChapterProcessState.Error -> FailedIndicator(enabled, modifier, onClick)
    }
}

@Composable
private fun IdleIndicator(
    enabled: Boolean,
    modifier: Modifier,
    onClick: (ChapterProcessAction) -> Unit,
) {
    Box(
        modifier = modifier
            .size(IconButtonTokens.StateLayerSize)
            .processClickable(
                enabled = enabled,
                hapticFeedback = LocalHapticFeedback.current,
                onLongClick = { onClick(ChapterProcessAction.START) },
                onClick = { onClick(ChapterProcessAction.START) },
            )
            .secondaryItemAlpha(),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            imageVector = MaterialSymbols.Rounded.Palette,
            contentDescription = "Process downloaded pages",
            modifier = Modifier.size(IndicatorSize),
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/** [progress] null means waiting in the queue: a ring that turns without a value. */
@Composable
private fun WorkingIndicator(
    enabled: Boolean,
    progress: Float?,
    modifier: Modifier,
    onClick: (ChapterProcessAction) -> Unit,
) {
    var isMenuExpanded by remember { mutableStateOf(false) }
    Box(
        modifier = modifier
            .size(IconButtonTokens.StateLayerSize)
            .processClickable(
                enabled = enabled,
                hapticFeedback = LocalHapticFeedback.current,
                onLongClick = { onClick(ChapterProcessAction.CANCEL) },
                onClick = { isMenuExpanded = true },
            ),
        contentAlignment = Alignment.Center,
    ) {
        val color = MaterialTheme.colorScheme.onSurfaceVariant
        val ringModifier = Modifier
            .size(IndicatorSize)
            .padding(IndicatorPadding)
        if (progress == null) {
            CircularProgressIndicator(
                modifier = ringModifier,
                color = color,
                strokeWidth = IndicatorPadding,
                trackColor = Color.Transparent,
                strokeCap = StrokeCap.Butt,
            )
        } else {
            CircularProgressIndicator(
                progress = { progress.coerceIn(0f, 1f) },
                modifier = ringModifier,
                color = color,
                strokeWidth = IndicatorPadding,
                trackColor = Color.Transparent,
                strokeCap = StrokeCap.Butt,
            )
        }
        Icon(
            imageVector = MaterialSymbols.Rounded.Palette,
            contentDescription = null,
            modifier = Modifier.size(IndicatorSize - 8.dp),
            tint = color,
        )
        DropdownMenu(expanded = isMenuExpanded, onDismissRequest = { isMenuExpanded = false }) {
            DropdownMenuItem(
                text = { Text(text = "Cancel") },
                onClick = {
                    onClick(ChapterProcessAction.CANCEL)
                    isMenuExpanded = false
                },
            )
        }
    }
}

@Composable
private fun DoneIndicator(
    enabled: Boolean,
    modifier: Modifier,
    onClick: (ChapterProcessAction) -> Unit,
) {
    var isMenuExpanded by remember { mutableStateOf(false) }
    Box(
        modifier = modifier
            .size(IconButtonTokens.StateLayerSize)
            .processClickable(
                enabled = enabled,
                hapticFeedback = LocalHapticFeedback.current,
                onLongClick = { isMenuExpanded = true },
                onClick = { isMenuExpanded = true },
            ),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            imageVector = MaterialSymbols.Rounded.Palette,
            contentDescription = "Processed",
            modifier = Modifier.size(IndicatorSize),
            tint = MaterialTheme.colorScheme.primary,
        )
        DropdownMenu(expanded = isMenuExpanded, onDismissRequest = { isMenuExpanded = false }) {
            DropdownMenuItem(
                text = { Text(text = "Process again") },
                onClick = {
                    onClick(ChapterProcessAction.START)
                    isMenuExpanded = false
                },
            )
            DropdownMenuItem(
                text = { Text(text = "Delete processed pages") },
                onClick = {
                    onClick(ChapterProcessAction.DELETE)
                    isMenuExpanded = false
                },
            )
        }
    }
}

@Composable
private fun FailedIndicator(
    enabled: Boolean,
    modifier: Modifier,
    onClick: (ChapterProcessAction) -> Unit,
) {
    Box(
        modifier = modifier
            .size(IconButtonTokens.StateLayerSize)
            .processClickable(
                enabled = enabled,
                hapticFeedback = LocalHapticFeedback.current,
                onLongClick = { onClick(ChapterProcessAction.START) },
                onClick = { onClick(ChapterProcessAction.START) },
            ),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            imageVector = MaterialSymbols.Rounded.Info,
            contentDescription = "Processing failed, tap to try again",
            modifier = Modifier.size(IndicatorSize),
            tint = MaterialTheme.colorScheme.error,
        )
    }
}

private fun Modifier.processClickable(
    enabled: Boolean,
    hapticFeedback: HapticFeedback,
    onLongClick: () -> Unit,
    onClick: () -> Unit,
) = this.combinedClickable(
    enabled = enabled,
    onLongClick = {
        onLongClick()
        hapticFeedback.performHapticFeedback(HapticFeedbackType.LongPress)
    },
    onClick = onClick,
    role = Role.Button,
    interactionSource = null,
    indication = ripple(
        bounded = false,
        radius = IconButtonTokens.StateLayerSize / 2,
    ),
)

private val IndicatorSize = 23.dp
private val IndicatorPadding = 2.dp
