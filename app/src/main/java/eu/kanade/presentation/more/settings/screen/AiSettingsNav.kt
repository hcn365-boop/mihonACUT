package eu.kanade.presentation.more.settings.screen

/**
 * Tells the AI settings screen which tab to open first, for a button somewhere else in the app (the manga
 * screen) that goes straight to one tab. The value is used once: the next time the screen opens it starts on
 * the first tab again.
 */
object AiSettingsNav {

    /** The position of the "Batch" tab in the tab list of the AI settings screen. */
    const val BATCH = 4

    @Volatile
    var initialTab: Int = 0

    fun takeInitialTab(): Int = initialTab.also { initialTab = 0 }
}
