package eu.kanade.tachiyomi.ui.reader.colorize

/** Where a downloaded chapter is in the post-download processing (colorize, upscale, translate). */
sealed interface ChapterProcessState {

    /** Nothing was done to this chapter yet. */
    data object NotProcessed : ChapterProcessState

    /** Waiting for its turn in the queue. */
    data object Queued : ChapterProcessState

    /** Being processed; [progress] is from 0 to 1 over the pages of the chapter. */
    data class Processing(val progress: Float) : ChapterProcessState

    /** Every page was processed and saved. */
    data object Processed : ChapterProcessState

    /** Stopped by an error; pressing the button tries again. */
    data object Error : ChapterProcessState
}
