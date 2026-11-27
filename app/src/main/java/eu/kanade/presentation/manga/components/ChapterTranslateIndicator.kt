package eu.kanade.presentation.manga.components

import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import eu.kanade.presentation.components.DropdownMenu
import eu.kanade.tachiyomi.ui.reader.colorize.ChapterProcessState
import tachiyomi.presentation.core.components.material.IconButtonTokens
import tachiyomi.presentation.core.util.secondaryItemAlpha

/**
 * The translation button of a chapter, next to the processing button. It translates the downloaded pages
 * alone, with the Batch settings, without touching the images: for a chapter only to be translated in advance,
 * or to try the translation again after it failed during the full processing. The actions are the same as
 * the ones of [ChapterProcessIndicator].
 */
@Composable
fun ChapterTranslateIndicator(
    enabled: Boolean,
    stateProvider: () -> ChapterProcessState,
    onClick: (ChapterProcessAction) -> Unit,
    modifier: Modifier = Modifier,
) {
    when (val state = stateProvider()) {
        ChapterProcessState.NotProcessed, ChapterProcessState.ProcessedFinal -> Idle(enabled, modifier, onClick)
        ChapterProcessState.Queued -> Working(enabled, null, modifier, onClick)
        is ChapterProcessState.Processing -> Working(enabled, state.progress, modifier, onClick)
        ChapterProcessState.Processed -> Done(enabled, modifier, onClick)
        ChapterProcessState.Error -> Failed(enabled, modifier, onClick)
    }
}

/** The mark of the button: two letters, a Latin one and a Japanese one. */
@Composable
private fun Mark(color: Color, size: Int = 12) {
    Text(
        text = "A文",
        color = color,
        fontSize = size.sp,
        fontWeight = FontWeight.Bold,
        maxLines = 1,
    )
}

@Composable
private fun Idle(enabled: Boolean, modifier: Modifier, onClick: (ChapterProcessAction) -> Unit) {
    Box(
        modifier = modifier
            .size(IconButtonTokens.StateLayerSize)
            .translateClickable(
                enabled = enabled,
                hapticFeedback = LocalHapticFeedback.current,
                onLongClick = { onClick(ChapterProcessAction.START) },
                onClick = { onClick(ChapterProcessAction.START) },
            )
            .secondaryItemAlpha(),
        contentAlignment = Alignment.Center,
    ) {
        Mark(MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** [progress] null means waiting in the queue: a ring that turns without a value. */
@Composable
private fun Working(
    enabled: Boolean,
    progress: Float?,
    modifier: Modifier,
    onClick: (ChapterProcessAction) -> Unit,
) {
    var isMenuExpanded by remember { mutableStateOf(false) }
    Box(
        modifier = modifier
            .size(IconButtonTokens.StateLayerSize)
            .translateClickable(
                enabled = enabled,
                hapticFeedback = LocalHapticFeedback.current,
                onLongClick = { onClick(ChapterProcessAction.CANCEL) },
                onClick = { isMenuExpanded = true },
            ),
        contentAlignment = Alignment.Center,
    ) {
        val color = MaterialTheme.colorScheme.onSurfaceVariant
        val ring = Modifier
            .size(RingSize)
            .padding(RingPadding)
        if (progress == null) {
            CircularProgressIndicator(
                modifier = ring,
                color = color,
                strokeWidth = RingPadding,
                trackColor = Color.Transparent,
                strokeCap = StrokeCap.Butt,
            )
        } else {
            CircularProgressIndicator(
                progress = { progress.coerceIn(0f, 1f) },
                modifier = ring,
                color = color,
                strokeWidth = RingPadding,
                trackColor = Color.Transparent,
                strokeCap = StrokeCap.Butt,
            )
        }
        Mark(color, size = 9)
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
private fun Done(enabled: Boolean, modifier: Modifier, onClick: (ChapterProcessAction) -> Unit) {
    var isMenuExpanded by remember { mutableStateOf(false) }
    Box(
        modifier = modifier
            .size(IconButtonTokens.StateLayerSize)
            .translateClickable(
                enabled = enabled,
                hapticFeedback = LocalHapticFeedback.current,
                onLongClick = { isMenuExpanded = true },
                onClick = { isMenuExpanded = true },
            ),
        contentAlignment = Alignment.Center,
    ) {
        Mark(MaterialTheme.colorScheme.primary)
        DropdownMenu(expanded = isMenuExpanded, onDismissRequest = { isMenuExpanded = false }) {
            DropdownMenuItem(
                text = { Text(text = "Translate again") },
                onClick = {
                    onClick(ChapterProcessAction.START)
                    isMenuExpanded = false
                },
            )
            DropdownMenuItem(
                text = { Text(text = "Delete the translations of this chapter") },
                onClick = {
                    onClick(ChapterProcessAction.DELETE)
                    isMenuExpanded = false
                },
            )
        }
    }
}

@Composable
private fun Failed(enabled: Boolean, modifier: Modifier, onClick: (ChapterProcessAction) -> Unit) {
    Box(
        modifier = modifier
            .size(IconButtonTokens.StateLayerSize)
            .translateClickable(
                enabled = enabled,
                hapticFeedback = LocalHapticFeedback.current,
                onLongClick = { onClick(ChapterProcessAction.START) },
                onClick = { onClick(ChapterProcessAction.START) },
            ),
        contentAlignment = Alignment.Center,
    ) {
        Mark(MaterialTheme.colorScheme.error)
    }
}

private fun Modifier.translateClickable(
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

private val RingSize = 23.dp
private val RingPadding = 2.dp
