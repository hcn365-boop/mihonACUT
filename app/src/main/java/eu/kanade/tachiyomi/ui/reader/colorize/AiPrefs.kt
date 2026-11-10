package eu.kanade.tachiyomi.ui.reader.colorize

import eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences
import tachiyomi.core.common.preference.Preference

/**
 * The AI settings, in one of three sets: the general ones, the copy that belongs to one manga ([mangaId]),
 * or the copy used for post-download processing ([scope] "_post").
 *
 * Settings that describe how a manga should look (colorize, upscale, translation, language...) can be
 * kept per manga. Settings that belong to the phone or to a service account (memory, cache, threads,
 * hardware, model links, API key, server, request limits) are always the general ones.
 */
class AiPrefs(
    private val base: ReaderPreferences,
    private val mangaId: Long?,
    scope: String? = null,
) {

    /** Added to the key of every setting that can be kept apart; null for the general settings. */
    private val suffix: String? = scope ?: mangaId?.let { "_m$it" }

    private fun <T> own(pref: Preference<T>): Preference<T> =
        if (suffix != null) base.scoped(pref, suffix) else pref

    // region Can belong to one manga

    val colorizeEnabled: Preference<Boolean> = own(base.colorizeEnabled)
    val colorizeIntensity: Preference<Int> = own(base.colorizeIntensity)
    val colorizeMaxMegapixels: Preference<Int> = own(base.colorizeMaxMegapixels)
    val colorizePreloadCount: Preference<Int> = own(base.colorizePreloadCount)
    val colorizeUseSam: Preference<Boolean> = own(base.colorizeUseSam)
    val colorizeSaturationEnabled: Preference<Boolean> = own(base.colorizeSaturationEnabled)
    val colorizeSaturation: Preference<Int> = own(base.colorizeSaturation)

    val upscaleEnabled: Preference<Boolean> = own(base.upscaleEnabled)
    val upscaleRestoreModel: Preference<String> = own(base.upscaleRestoreModel)
    val upscaleMainModel: Preference<String> = own(base.upscaleMainModel)
    val upscaleMaxMegapixels: Preference<Int> = own(base.upscaleMaxMegapixels)
    val upscaleMaxOutputMegapixels: Preference<Int> = own(base.upscaleMaxOutputMegapixels)
    val upscaleNormalizeEnabled: Preference<Boolean> = own(base.upscaleNormalizeEnabled)
    val upscaleNormalizeShortSide: Preference<Int> = own(base.upscaleNormalizeShortSide)
    val upscalePreloadCount: Preference<Int> = own(base.upscalePreloadCount)
    val upscaleFinalPassEnabled: Preference<Boolean> = own(base.upscaleFinalPassEnabled)
    val upscaleFinalPassKeepSize: Preference<Boolean> = own(base.upscaleFinalPassKeepSize)
    val upscaleFinalPassMaxMegapixels: Preference<Int> = own(base.upscaleFinalPassMaxMegapixels)

    val translateEnabled: Preference<Boolean> = own(base.translateEnabled)
    val translateFrom: Preference<String> = own(base.translateFrom)
    val translateTo: Preference<String> = own(base.translateTo)
    val translationEngine: Preference<String> = own(base.translationEngine)
    val translationOcrMode: Preference<String> = own(base.translationOcrMode)
    val translationFont: Preference<String> = own(base.translationFont)
    val translatePreloadCount: Preference<Int> = own(base.translatePreloadCount)
    val translateAutoLanguage: Preference<Boolean> = own(base.translateAutoLanguage)
    val translateAutoFromTitle: Preference<Boolean> = own(base.translateAutoFromTitle)
    val translateAutoFromSource: Preference<Boolean> = own(base.translateAutoFromSource)
    val translateConfirmLanguage: Preference<Boolean> = own(base.translateConfirmLanguage)
    val translateCollisionPadding: Preference<Int> = own(base.translateCollisionPadding)
    val translateFilterWordsEnabled: Preference<Boolean> = own(base.translateFilterWordsEnabled)
    val translateFilterWords: Preference<String> = own(base.translateFilterWords)
    val translateLatinFilterEnabled: Preference<Boolean> = own(base.translateLatinFilterEnabled)
    val translateLatinMinLetters: Preference<Int> = own(base.translateLatinMinLetters)
    val translateSkipSameLanguage: Preference<Boolean> = own(base.translateSkipSameLanguage)
    val translateLatinKeepWords: Preference<String> = own(base.translateLatinKeepWords)
    val translateLatinKeepCommon: Preference<Boolean> = own(base.translateLatinKeepCommon)

    fun forceTranslate(mangaId: Long): Preference<Boolean> = base.mangaForceTranslate(mangaId)

    // endregion

    // region Always general

    val colorizeModelUrl: Preference<String> = base.colorizeModelUrl
    val upscaleModelsBaseUrl: Preference<String> = base.upscaleModelsBaseUrl
    val colorizeAcceleration: Preference<String> = base.colorizeAcceleration
    val upscaleAcceleration: Preference<String> = base.upscaleAcceleration
    val aiThreads: Preference<Int> = base.aiThreads
    val aiMemoryBudgetMb: Preference<Int> = base.aiMemoryBudgetMb
    val aiCacheSizeMb: Preference<Int> = base.aiCacheSizeMb
    val pagePreloadCount: Preference<Int> = base.pagePreloadCount
    val translationModel: Preference<String> = base.translationModel
    val translationApiKey: Preference<String> = base.translationApiKey
    val translationTemperature: Preference<String> = base.translationTemperature
    val translationMaxTokens: Preference<String> = base.translationMaxTokens
    val translationServerUrl: Preference<String> = base.translationServerUrl
    val translatePagesPerRequest: Preference<Int> = base.translatePagesPerRequest
    val translateMaxChars: Preference<Int> = base.translateMaxChars
    val translateMaxWaitSeconds: Preference<Int> = base.translateMaxWaitSeconds

    // endregion

    /** The translation settings; any change to one of them rebuilds the translation snapshot. */
    val translationPrefs: List<Preference<*>> = listOf(
        translateEnabled, translateFrom, translateTo, translationEngine, translationOcrMode, translationFont,
        translatePreloadCount, translateAutoLanguage, translateAutoFromTitle, translateAutoFromSource,
        translateConfirmLanguage, translationModel, translationApiKey, translationTemperature,
        translationMaxTokens, translationServerUrl, translatePagesPerRequest, translateMaxChars,
        translateMaxWaitSeconds, translateCollisionPadding, translateFilterWordsEnabled,
        translateFilterWords, translateLatinFilterEnabled, translateLatinMinLetters,
        translateSkipSameLanguage, translateLatinKeepWords, translateLatinKeepCommon,
    )

    /** Settings that change how a page looks: a change reloads the pages being read. */
    val appearancePrefs: List<Preference<*>> = listOf(
        colorizeEnabled, colorizeUseSam, colorizeIntensity, colorizeMaxMegapixels,
        colorizeSaturationEnabled, colorizeSaturation,
        upscaleEnabled, upscaleRestoreModel, upscaleMainModel, upscaleMaxMegapixels, upscaleMaxOutputMegapixels,
        upscaleNormalizeEnabled, upscaleNormalizeShortSide,
        upscaleFinalPassEnabled, upscaleFinalPassKeepSize, upscaleFinalPassMaxMegapixels,
    ) + translationPrefs

    /** Settings of the phone and of the queues: a change does not reload pages. */
    val runtimePrefs: List<Preference<*>> = listOf(
        aiMemoryBudgetMb, aiCacheSizeMb, pagePreloadCount, colorizePreloadCount, upscalePreloadCount,
        colorizeAcceleration, upscaleAcceleration, aiThreads,
    )

    /** Puts the settings of this scope back to their defaults. Keys, server addresses and model links are kept. */
    fun resetToDefaults() {
        fun <T> reset(pref: Preference<T>) = pref.set(pref.defaultValue())
        reset(colorizeEnabled)
        reset(colorizeIntensity)
        reset(colorizeMaxMegapixels)
        reset(colorizePreloadCount)
        reset(colorizeUseSam)
        reset(colorizeSaturationEnabled)
        reset(colorizeSaturation)
        reset(upscaleEnabled)
        reset(upscaleRestoreModel)
        reset(upscaleMainModel)
        reset(upscaleMaxMegapixels)
        reset(upscaleMaxOutputMegapixels)
        reset(upscaleNormalizeEnabled)
        reset(upscaleNormalizeShortSide)
        reset(upscalePreloadCount)
        reset(upscaleFinalPassEnabled)
        reset(upscaleFinalPassKeepSize)
        reset(upscaleFinalPassMaxMegapixels)
        reset(translateEnabled)
        reset(translateFrom)
        reset(translateTo)
        reset(translationEngine)
        reset(translationOcrMode)
        reset(translationFont)
        reset(translatePreloadCount)
        reset(translateAutoLanguage)
        reset(translateAutoFromTitle)
        reset(translateAutoFromSource)
        reset(translateConfirmLanguage)
        reset(colorizeAcceleration)
        reset(upscaleAcceleration)
        reset(aiThreads)
        reset(aiMemoryBudgetMb)
        reset(aiCacheSizeMb)
        reset(pagePreloadCount)
        reset(translationTemperature)
        reset(translationMaxTokens)
        reset(translatePagesPerRequest)
        reset(translateMaxChars)
        reset(translateMaxWaitSeconds)
        reset(translateCollisionPadding)
        reset(translateFilterWordsEnabled)
        reset(translateFilterWords)
        reset(translateLatinFilterEnabled)
        reset(translateLatinMinLetters)
        reset(translateSkipSameLanguage)
        reset(translateLatinKeepWords)
        reset(translateLatinKeepCommon)
    }
    
    /** Gives this set a copy of the general settings, so it starts from what you have now. */
    fun copyGeneralSettings() {
        val key = suffix ?: return
        fun <T> copy(general: Preference<T>) = base.scoped(general, key).set(general.get())
        copy(base.colorizeEnabled)
        copy(base.colorizeIntensity)
        copy(base.colorizeMaxMegapixels)
        copy(base.colorizePreloadCount)
        copy(base.colorizeUseSam)
        copy(base.colorizeSaturationEnabled)
        copy(base.colorizeSaturation)
        copy(base.upscaleEnabled)
        copy(base.upscaleRestoreModel)
        copy(base.upscaleMainModel)
        copy(base.upscaleMaxMegapixels)
        copy(base.upscaleMaxOutputMegapixels)
        copy(base.upscaleNormalizeEnabled)
        copy(base.upscaleNormalizeShortSide)
        copy(base.upscalePreloadCount)
        copy(base.upscaleFinalPassEnabled)
        copy(base.upscaleFinalPassKeepSize)
        copy(base.upscaleFinalPassMaxMegapixels)
        copy(base.translateEnabled)
        copy(base.translateFrom)
        copy(base.translateTo)
        copy(base.translationEngine)
        copy(base.translationOcrMode)
        copy(base.translationFont)
        copy(base.translatePreloadCount)
        copy(base.translateAutoLanguage)
        copy(base.translateAutoFromTitle)
        copy(base.translateAutoFromSource)
        copy(base.translateConfirmLanguage)
        copy(base.translateCollisionPadding)
        copy(base.translateFilterWordsEnabled)
        copy(base.translateFilterWords)
        copy(base.translateLatinFilterEnabled)
        copy(base.translateLatinMinLetters)
        copy(base.translateSkipSameLanguage)
        copy(base.translateLatinKeepWords)
        copy(base.translateLatinKeepCommon)
    }
}
