package eu.kanade.tachiyomi.ui.reader.colorize

import eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences
import eu.kanade.translation.AutoLanguage
import eu.kanade.translation.AutoLanguageOptions
import eu.kanade.translation.BatchLimits
import eu.kanade.translation.MangaTranslationContext
import eu.kanade.translation.OcrMode
import eu.kanade.translation.PageFilterOptions
import eu.kanade.translation.TranslationFont
import eu.kanade.translation.recognizer.TextRecognizerLanguage
import eu.kanade.translation.translator.TextTranslatorLanguage
import eu.kanade.translation.translator.TextTranslators
import eu.kanade.translation.translator.TranslatorSettings
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asContextElement
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import tachiyomi.core.common.preference.Preference

/** Everything the translation stage needs, as one immutable snapshot. Null in the snapshot when it is off. */
class TranslationConfig(
    val settings: TranslatorSettings,
    val ocrMode: OcrMode,
    val font: TranslationFont,
    val limits: BatchLimits,
    /** How many pages ahead are read (OCR) and queued for translation in the background. */
    val preload: Int,
    val auto: AutoLanguageOptions,
    val filter: PageFilterOptions,
) {
    /** The settings with the page language chosen for the manga being processed (when auto language is on). */
    fun effectiveSettings(): TranslatorSettings {
        val language = AutoLanguage.resolve(
            settings.from,
            auto,
            null,
            AiConfig.mangaTitle,
            AiConfig.mangaSourceLang,
        ).language
        return if (language == settings.from) settings else settings.copy(from = language)
    }
}

/** All the AI settings at one moment. Never changes after it is made. */
class AiSnapshot(
    val colorizeEnabled: Boolean = false,
    val colorizeUseSam: Boolean = false,
    /** 0 = no color, 1 = model color, up to 1.2 */
    val colorizeIntensity: Float = 1f,
    val colorizeMaxPixels: Long = 12_000_000L,
    val upscaleEnabled: Boolean = false,
    val upscaleRestoreModel: String = "",
    val upscaleMainModel: String = "Upscale_L",
    val upscaleFinalPassEnabled: Boolean = false,
    val upscaleFinalPassKeepSize: Boolean = true,
    val upscaleFinalPassMaxPixels: Long = 4_000_000L,
    val upscaleMaxPixels: Long = 3_000_000L,
    /** Upscaled pages bigger than this are scaled down: the viewer needs a full buffer per page. */
    val upscaleMaxOutputPixels: Long = 12_000_000L,
    val upscaleNormalizeEnabled: Boolean = false,
    val upscaleNormalizeSide: Int = 1080,
    /** Estimated memory one page may need while being processed. Bigger pages are left as they are. */
    val memoryBudgetBytes: Long = 1024L * 1024 * 1024,
    /** Disk cache of processed pages. */
    val cacheBytes: Long = 1024L * 1024 * 1024,
    /** Pages the loader downloads ahead of the current one. */
    val pagePreload: Int = 4,
    /** Pages ahead that are colorized / upscaled in the background (used by the preload queue). */
    val colorizePreload: Int = 3,
    val upscalePreload: Int = 2,
    val colorizeAcceleration: Acceleration = Acceleration.CPU,
    val upscaleAcceleration: Acceleration = Acceleration.CPU,
    /** CPU threads for the AI models; 0 lets ONNX Runtime decide. */
    val aiThreads: Int = 0,
    /** Translation settings, or null when translation is off. */
    val translation: TranslationConfig? = null,
    val translateCollisionPadding: Float = 1.3f,
    /** The manga this snapshot is for, when it is not the one in [MangaTranslationContext] (post-download jobs). */
    val mangaTitle: String? = null,
    val mangaSourceLang: String? = null,
    /** Null: ask the manga's own switch. Otherwise this answer (post-download jobs never force). */
    val forceTranslate: Boolean? = null,
)

/**
 * Current AI settings, readable from anywhere (page loaders, holders, background jobs) without
 * dependency injection. By default they are the live snapshot, which follows the manga being read:
 * the general settings, or the settings of that manga when its "separate settings" box is checked
 * ([AiPrefs] decides which).
 *
 * A job can run with another snapshot through [withSnapshot]: inside it every read below gives that
 * snapshot, and the reader keeps its own settings at the same time.
 */
object AiConfig {

    @Volatile
    private var live = AiSnapshot()

    private val override = ThreadLocal<AiSnapshot?>()

    private val s: AiSnapshot
        get() = override.get() ?: live

    /** True inside [withSnapshot]: a job that is not the reader. */
    val isPost: Boolean
        get() = override.get() != null

    /** Runs [block] (and everything it calls, on any thread) with [snapshot] as the settings. */
    suspend fun <T> withSnapshot(snapshot: AiSnapshot, block: suspend () -> T): T =
        withContext(override.asContextElement(snapshot)) { block() }

    val colorizeEnabled: Boolean get() = s.colorizeEnabled
    val colorizeUseSam: Boolean get() = s.colorizeUseSam
    val colorizeIntensity: Float get() = s.colorizeIntensity
    val colorizeMaxPixels: Long get() = s.colorizeMaxPixels
    val upscaleEnabled: Boolean get() = s.upscaleEnabled
    val upscaleRestoreModel: String get() = s.upscaleRestoreModel
    val upscaleMainModel: String get() = s.upscaleMainModel
    val upscaleFinalPassEnabled: Boolean get() = s.upscaleFinalPassEnabled
    val upscaleFinalPassKeepSize: Boolean get() = s.upscaleFinalPassKeepSize
    val upscaleFinalPassMaxPixels: Long get() = s.upscaleFinalPassMaxPixels
    val upscaleMaxPixels: Long get() = s.upscaleMaxPixels
    val upscaleMaxOutputPixels: Long get() = s.upscaleMaxOutputPixels
    val upscaleNormalizeEnabled: Boolean get() = s.upscaleNormalizeEnabled
    val upscaleNormalizeSide: Int get() = s.upscaleNormalizeSide
    val memoryBudgetBytes: Long get() = s.memoryBudgetBytes
    val cacheBytes: Long get() = s.cacheBytes
    val pagePreload: Int get() = s.pagePreload
    val colorizePreload: Int get() = s.colorizePreload
    val upscalePreload: Int get() = s.upscalePreload
    val colorizeAcceleration: Acceleration get() = s.colorizeAcceleration
    val upscaleAcceleration: Acceleration get() = s.upscaleAcceleration
    val aiThreads: Int get() = s.aiThreads
    val translation: TranslationConfig? get() = s.translation
    val translateCollisionPadding: Float get() = s.translateCollisionPadding

    /** Short side (pixels) big pages are shrunk to before upscaling, or 0 when this is off. */
    val normalizeSide: Int
        get() = if (s.upscaleNormalizeEnabled) s.upscaleNormalizeSide else 0

    /** The manga being processed: the one of the snapshot, or the one the reader has open. */
    val mangaTitle: String
        get() = s.mangaTitle ?: MangaTranslationContext.title

    val mangaSourceLang: String
        get() = s.mangaSourceLang ?: MangaTranslationContext.sourceLang

    val anyEnabled: Boolean
        get() = s.colorizeEnabled || s.upscaleEnabled || s.upscaleNormalizeEnabled ||
            s.upscaleFinalPassEnabled || s.translation != null

    private var prefsRef: ReaderPreferences? = null
    private var viewerScope: CoroutineScope? = null
    private var onOutputChanged: (() -> Unit)? = null
    private var bindJob: Job? = null
    private var notifyJob: Job? = null

    /** True when the user asked to translate this manga even if the page language is the target language. */
    fun forceTranslate(mangaId: Long): Boolean =
        override.get()?.forceTranslate
            ?: (mangaId >= 0 && prefsRef?.mangaForceTranslate(mangaId)?.get() == true)

    /**
     * The snapshot of the post-download settings, for a job that processes downloaded chapters.
     * [mangaTitle] and [sourceLang] belong to the manga being processed (used by automatic language).
     */
    fun postSnapshot(prefs: ReaderPreferences, mangaTitle: String, sourceLang: String): AiSnapshot =
        snapshotFrom(AiPrefs(prefs, null, "_post"), mangaTitle, sourceLang, forceTranslate = false)

    /**
     * Copies the preferences now and whenever they change.
     * [onOutputChanged] runs after a change that alters how pages look, so the viewer can reload them.
     */
    fun attach(prefs: ReaderPreferences, scope: CoroutineScope, onOutputChanged: () -> Unit) {
        prefsRef = prefs
        viewerScope = scope
        this.onOutputChanged = onOutputChanged
        bindAll(notify = false)
    }

    /** Call when the manga being read changes, so the right set of settings is used. */
    fun onMangaChanged() {
        bindAll(notify = false)
    }

    /** Several changes can come together, and reloading the pages while images are opening can crash the viewer. */
    private fun notifyChanged() {
        val scope = viewerScope ?: return
        notifyJob?.cancel()
        notifyJob = scope.launch {
            delay(600)
            onOutputChanged?.invoke()
        }
    }

    private fun bindAll(notify: Boolean) {
        val prefs = prefsRef ?: return
        val scope = viewerScope ?: return
        bindJob?.cancel()
        val job = SupervisorJob(scope.coroutineContext[Job])
        bindJob = job
        val bindScope = CoroutineScope(scope.coroutineContext + job)

        val mangaId = MangaTranslationContext.mangaId
        val own = mangaId >= 0 && prefs.mangaOverride(mangaId).get()
        val p = AiPrefs(prefs, if (own) mangaId else null)

        // Every setting rebuilds the whole snapshot, so it is never half updated.
        live = snapshotFrom(p)

        // The first value of a flow is not a change, and equal values are not changes either.
        fun watch(list: List<Preference<*>>, affectsOutput: Boolean) {
            list.forEach { pref ->
                pref.changes()
                    .distinctUntilChanged()
                    .drop(1)
                    .onEach {
                        live = snapshotFrom(p)
                        if (affectsOutput) notifyChanged()
                    }
                    .launchIn(bindScope)
            }
        }
        watch(p.appearancePrefs, affectsOutput = true)
        watch(p.runtimePrefs, affectsOutput = false)

        // The box itself: checking or unchecking it switches every setting to the right set.
        if (mangaId >= 0) {
            prefs.mangaOverride(mangaId).changes()
                .distinctUntilChanged()
                .drop(1)
                .onEach { bindAll(notify = true) }
                .launchIn(bindScope)

            prefs.mangaForceTranslate(mangaId).changes()
                .distinctUntilChanged()
                .drop(1)
                .onEach { notifyChanged() }
                .launchIn(bindScope)
        }

        if (notify) notifyChanged()
    }

    private fun snapshotFrom(
        p: AiPrefs,
        mangaTitle: String? = null,
        sourceLang: String? = null,
        forceTranslate: Boolean? = null,
    ): AiSnapshot = AiSnapshot(
        colorizeEnabled = p.colorizeEnabled.get(),
        colorizeUseSam = p.colorizeUseSam.get(),
        colorizeIntensity = p.colorizeIntensity.get() / 100f,
        colorizeMaxPixels = p.colorizeMaxMegapixels.get() * 1_000_000L,
        upscaleEnabled = p.upscaleEnabled.get(),
        upscaleRestoreModel = p.upscaleRestoreModel.get(),
        upscaleMainModel = p.upscaleMainModel.get(),
        upscaleFinalPassEnabled = p.upscaleFinalPassEnabled.get(),
        upscaleFinalPassKeepSize = p.upscaleFinalPassKeepSize.get(),
        upscaleFinalPassMaxPixels = p.upscaleFinalPassMaxMegapixels.get() * 1_000_000L,
        upscaleMaxPixels = p.upscaleMaxMegapixels.get() * 1_000_000L,
        upscaleMaxOutputPixels = p.upscaleMaxOutputMegapixels.get() * 1_000_000L,
        upscaleNormalizeEnabled = p.upscaleNormalizeEnabled.get(),
        upscaleNormalizeSide = p.upscaleNormalizeShortSide.get(),
        memoryBudgetBytes = p.aiMemoryBudgetMb.get() * 1024L * 1024,
        cacheBytes = p.aiCacheSizeMb.get() * 1024L * 1024,
        pagePreload = p.pagePreloadCount.get(),
        colorizePreload = p.colorizePreloadCount.get(),
        upscalePreload = p.upscalePreloadCount.get(),
        colorizeAcceleration = Acceleration.fromName(p.colorizeAcceleration.get()),
        upscaleAcceleration = Acceleration.fromName(p.upscaleAcceleration.get()),
        aiThreads = p.aiThreads.get(),
        translation = buildTranslation(p),
        translateCollisionPadding = p.translateCollisionPadding.get() / 100f,
        mangaTitle = mangaTitle,
        mangaSourceLang = sourceLang,
        forceTranslate = forceTranslate,
    )

    private fun buildTranslation(p: AiPrefs): TranslationConfig? {
        if (!p.translateEnabled.get()) return null
        val from = p.translateFrom.get()
        val to = p.translateTo.get()
        return TranslationConfig(
            settings = TranslatorSettings(
                engine = TextTranslators.fromName(p.translationEngine.get()),
                from = TextRecognizerLanguage.entries.firstOrNull { it.name.equals(from, ignoreCase = true) }
                    ?: TextRecognizerLanguage.entries.first(),
                to = TextTranslatorLanguage.entries.firstOrNull { it.name.equals(to, ignoreCase = true) }
                    ?: TextTranslatorLanguage.entries.first(),
                model = p.translationModel.get(),
                apiKey = p.translationApiKey.get(),
                temperature = p.translationTemperature.get().toFloatOrNull() ?: 1f,
                maxOutputTokens = p.translationMaxTokens.get().toIntOrNull() ?: 8192,
                serverUrl = p.translationServerUrl.get(),
            ),
            ocrMode = OcrMode.entries.firstOrNull { it.name.equals(p.translationOcrMode.get(), ignoreCase = true) }
                ?: OcrMode.MLKIT,
            font = TranslationFont.fromName(p.translationFont.get()),
            limits = BatchLimits(
                maxPages = p.translatePagesPerRequest.get().coerceAtLeast(1),
                maxChars = p.translateMaxChars.get().coerceAtLeast(200),
                maxWaitMillis = p.translateMaxWaitSeconds.get().coerceAtLeast(1) * 1000L,
            ),
            preload = p.translatePreloadCount.get(),
            auto = AutoLanguageOptions(
                enabled = p.translateAutoLanguage.get(),
                fromTitle = p.translateAutoFromTitle.get(),
                fromSource = p.translateAutoFromSource.get(),
                confirmPerPage = p.translateConfirmLanguage.get(),
            ),
            filter = PageFilterOptions(
                words = if (p.translateFilterWordsEnabled.get()) {
                    PageFilterOptions.parseWords(p.translateFilterWords.get())
                } else {
                    emptyList()
                },
                latinEnabled = p.translateLatinFilterEnabled.get(),
                latinMinLetters = p.translateLatinMinLetters.get().coerceIn(1, 10),
                keepWords = PageFilterOptions.parseWords(p.translateLatinKeepWords.get()),
                keepCommon = p.translateLatinKeepCommon.get(),
                skipSameLanguage = p.translateSkipSameLanguage.get(),
            ),
        )
    }
}
