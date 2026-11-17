package eu.kanade.tachiyomi.ui.reader.colorize

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import com.hippo.unifile.UniFile
import eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences
import eu.kanade.translation.PageTranslationService
import eu.kanade.translation.TranslationStore
import eu.kanade.translation.model.PageTranslation
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import logcat.LogPriority
import okio.Buffer
import tachiyomi.core.common.util.system.ImageUtil
import tachiyomi.core.common.util.system.logcat
import java.util.concurrent.ConcurrentHashMap
import kotlin.coroutines.cancellation.CancellationException

/**
 * Processes downloaded chapters in advance, one chapter at a time, with the batch settings (the "Batch"
 * tab) and not the real-time ones.
 *
 * Three independent options (settings of the Batch tab):
 * - keep the original pages: the processed pages go to a "processed" folder next to them. When it is off, the
 *   processed page replaces the original (same name), and nothing can be undone;
 * - draw the translation into the images: permanent. When it is off, the translation is kept apart and drawn
 *   when the page is read;
 * - where the separate translation is kept: in the app's storage, or as json files in the chapter folder.
 *
 * The images never wait for the translator: the text of many pages is collected and sent together while
 * the next images are being processed.
 */
object PostProcessManager {

    /** One chapter to process: [chapterDir] is its download folder, whose image files are the pages. */
    class Request(
        val chapterId: Long,
        val mangaId: Long,
        val mangaTitle: String,
        val sourceLang: String,
        val chapterDir: UniFile,
    )

    private class Entry(val context: Context, val prefs: ReaderPreferences, val request: Request)

    /** What is known about one page while its chapter is processed. */
    private class PageRecord(
        val baseName: String,
        val originalUri: Uri,
        val originalHash: String,
        /** The fingerprint of the bytes the reader will read for this page. */
        val readHash: String,
        /** Where the image is after the image steps: the original itself when nothing changed. */
        val cleanUri: Uri,
    )

    private const val PROCESSED = "processed"

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val queue = Channel<Entry>(Channel.UNLIMITED)
    private val cancelled = ConcurrentHashMap.newKeySet<Long>()
    private val lock = Any()
    private var worker: Job? = null

    @Volatile
    private var runningId = -1L

    @Volatile
    private var runningJob: Job? = null

    @Volatile
    private var loaded = false

    private val _states = MutableStateFlow<Map<Long, ChapterProcessState>>(emptyMap())

    /** The state of every chapter that is queued, running, done or failed. Others are not processed. */
    val states: StateFlow<Map<Long, ChapterProcessState>> = _states.asStateFlow()

    private fun setState(chapterId: Long, state: ChapterProcessState) {
        _states.update { current ->
            if (state == ChapterProcessState.NotProcessed) current - chapterId else current + (chapterId to state)
        }
    }

    /** Loads the chapters that were processed in earlier runs of the app. Safe to call often. */
    fun ensureLoaded(context: Context) {
        if (loaded) return
        loaded = true
        scope.launch {
            PostProcessStore.refresh(context)
            val done = PostProcessStore.chaptersWithState(context, PostProcessStore.STATE_PROCESSED)
            val final = PostProcessStore.chaptersWithState(context, PostProcessStore.STATE_FINAL)
            _states.update { current ->
                current +
                    done.filter { it !in current }.associateWith { ChapterProcessState.Processed } +
                    final.filter { it !in current }.associateWith { ChapterProcessState.ProcessedFinal }
            }
        }
    }

    fun enqueue(context: Context, prefs: ReaderPreferences, request: Request) {
        val id = request.chapterId
        when (_states.value[id]) {
            is ChapterProcessState.Processing, ChapterProcessState.Queued -> return
            else -> Unit
        }
        // Cancelled a moment ago and its entry is still in the queue: let that entry run again.
        if (cancelled.remove(id)) {
            setState(id, ChapterProcessState.Queued)
            return
        }
        setState(id, ChapterProcessState.Queued)
        queue.trySend(Entry(context.applicationContext, prefs, request))
        startWorker()
    }

    /** Shows the error state of a chapter that could not even be queued (for example a CBZ file). */
    fun fail(chapterId: Long) {
        setState(chapterId, ChapterProcessState.Error)
    }

    fun cancel(chapterId: Long) {
        if (runningId == chapterId) {
            runningJob?.cancel()
        } else if (_states.value[chapterId] == ChapterProcessState.Queued) {
            cancelled.add(chapterId)
            setState(chapterId, ChapterProcessState.NotProcessed)
        }
    }

    /** Deletes the processed pages and the translation files of a chapter. The original pages stay. */
    fun delete(context: Context, chapterId: Long) {
        // When the originals were replaced there is nothing separate to delete, and nothing can be undone.
        if (_states.value[chapterId] == ChapterProcessState.ProcessedFinal) return
        scope.launch {
            clearResults(context.applicationContext, chapterId)
            setState(chapterId, ChapterProcessState.NotProcessed)
        }
    }

    /** The downloaded image files of a chapter folder, in reading order. */
    fun listPages(chapterDir: UniFile): List<Uri> =
        chapterDir.listFiles().orEmpty()
            .filter { it.isFile && ImageUtil.isImage(it.name) { it.openInputStream() } }
            .sortedBy { it.name }
            .map { it.uri }

    private fun startWorker() {
        synchronized(lock) {
            if (worker?.isActive == true) return
            worker = scope.launch {
                for (entry in queue) {
                    if (cancelled.remove(entry.request.chapterId)) continue
                    run(entry)
                }
            }
        }
    }

    private suspend fun run(entry: Entry) {
        val id = entry.request.chapterId
        val context = entry.context
        val keepOriginals = entry.prefs.postKeepOriginals.get()
        val job = scope.launch {
            try {
                setState(id, ChapterProcessState.Processing(0f))
                process(entry)
                if (keepOriginals) {
                    PostProcessStore.setChapter(context, id, PostProcessStore.STATE_PROCESSED)
                    setState(id, ChapterProcessState.Processed)
                } else {
                    PostProcessStore.setChapter(context, id, PostProcessStore.STATE_FINAL)
                    setState(id, ChapterProcessState.ProcessedFinal)
                }
                PostProcessStore.refresh(context)
            } catch (e: CancellationException) {
                // With the originals replaced, the pages already done stay done: they cannot be undone.
                if (keepOriginals) clearResults(context, id)
                setState(id, ChapterProcessState.NotProcessed)
                throw e
            } catch (e: Throwable) {
                logcat(LogPriority.ERROR, e)
                PostProcessStore.setChapter(context, id, PostProcessStore.STATE_ERROR)
                setState(id, ChapterProcessState.Error)
            }
        }
        runningId = id
        runningJob = job
        job.join()
        runningJob = null
        runningId = -1L
    }

    private suspend fun process(entry: Entry) {
        val request = entry.request
        val context = entry.context
        val keepOriginals = entry.prefs.postKeepOriginals.get()
        val inImages = entry.prefs.postTranslationInImages.get()
        val inFolder = !inImages && entry.prefs.postTranslationInFolder.get()

        // The folder comes from the download manager: a folder found from a page address has no parent.
        val chapterDir = request.chapterDir
        val pages = listPages(chapterDir)
        if (pages.isEmpty()) error("The chapter has no downloaded pages")

        // Start clean: earlier results may come from other settings.
        clearResults(context, request.chapterId)

        val snapshot = AiConfig.postSnapshot(entry.prefs, request.mangaTitle, request.sourceLang)
        AiConfig.withSnapshot(snapshot) {
            val translation = AiConfig.translation
            val records = ArrayList<PageRecord>()
            // Pages whose translation is on its way, with the translation they wait for.
            val waiting = ArrayList<Pair<PageRecord, Deferred<PageTranslation?>>>()
            val imageShare = if (translation != null) 0.9f else 1f

            pages.forEachIndexed { index, uri ->
                kotlin.coroutines.coroutineContext.ensureActive()
                val started = System.currentTimeMillis()
                val originalFile = UniFile.fromUri(context, uri)
                val base = (originalFile?.name ?: "page_$index").substringBeforeLast('.')
                val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                    ?: error("Cannot read the page $uri")
                val originalHash = PageColorizer.contentHash(bytes)

                // 1. The image steps. No text is drawn here, so the page is saved as soon as it is ready.
                val processed = PageColorizer.processImagesToBytes(context, Buffer().write(bytes))
                var readHash = originalHash
                var cleanUri = uri
                if (processed == null) {
                    PostProcessStore.savePage(context, request.chapterId, originalHash, "")
                } else if (keepOriginals) {
                    val file = chapterDir.subFolder(PROCESSED).writeFile(base + extensionOf(processed), processed)
                    PostProcessStore.savePage(context, request.chapterId, originalHash, file.uri.toString())
                    cleanUri = file.uri
                } else {
                    // The processed page takes the place of the original. The reader will read these new
                    // bytes, so their fingerprint is marked as "leave as it is".
                    originalFile?.delete()
                    val file = chapterDir.writeFile(base + extensionOf(processed), processed)
                    readHash = PageColorizer.contentHash(processed)
                    PostProcessStore.savePage(context, request.chapterId, readHash, "")
                    cleanUri = file.uri
                }
                val record = PageRecord(base, uri, originalHash, readHash, cleanUri)
                records.add(record)

                // 2. The text: read it now and give it to the translation service, which sends the text of
                // several pages together (the limits of the batch settings) and not one request per page.
                if (translation != null) {
                    var bitmap: Bitmap? = null
                    try {
                        val deferred = PageTranslationService.enqueueTracked(
                            context,
                            translation.effectiveSettings(),
                            translation.ocrMode,
                            originalHash,
                            translation.limits,
                        ) { BitmapFactory.decodeByteArray(bytes, 0, bytes.size).also { bitmap = it } }
                        waiting.add(record to deferred)
                    } finally {
                        bitmap?.recycle()
                    }
                }

                logcat(LogPriority.ERROR) {
                    "Batch page ${index + 1}/${pages.size}: ${System.currentTimeMillis() - started} ms, " +
                        if (processed == null) "image left as it is" else "image processed (${processed.size / 1024} KB)"
                }
                setState(request.chapterId, ChapterProcessState.Processing((index + 1f) / pages.size * imageShare))
            }

            // 3. Send what is still waiting, then wait for the translations of the chapter and put each one
            // where the options say.
            if (translation != null && waiting.isNotEmpty()) {
                PageTranslationService.flushNow(translation.limits)
                waiting.forEachIndexed { index, (record, deferred) ->
                    val page = deferred.await()
                    if (page != null && page.blocks.isNotEmpty()) {
                        when {
                            inImages -> burn(context, request, chapterDir, keepOriginals, record, page)
                            inFolder -> {
                                val file = PostTranslationFiles.write(
                                    chapterDir.subFolder(PostTranslationFiles.FOLDER),
                                    record.baseName + ".json",
                                    page,
                                )
                                PostProcessStore.setTranslationUri(context, record.readHash, file.uri.toString())
                            }
                            else -> TranslationStore.save(
                                context,
                                TranslationStore.Kind.TRANSLATED,
                                PostRead.translationName(record.readHash),
                                page,
                            )
                        }
                    }
                    setState(
                        request.chapterId,
                        ChapterProcessState.Processing(imageShare + (1f - imageShare) * (index + 1f) / waiting.size),
                    )
                }
            }
        }
    }

    /** Draws the translation into the processed image and saves it again: permanent. */
    private fun burn(
        context: Context,
        request: Request,
        chapterDir: UniFile,
        keepOriginals: Boolean,
        record: PageRecord,
        page: PageTranslation,
    ) {
        val clean = context.contentResolver.openInputStream(record.cleanUri)?.use { it.readBytes() } ?: return
        val drawn = PostRead.drawOn(context, clean, page) ?: return
        val name = record.baseName + extensionOf(drawn)
        if (keepOriginals) {
            // The earlier processed file may have another extension: it is replaced, never the original.
            if (record.cleanUri != record.originalUri) UniFile.fromUri(context, record.cleanUri)?.delete()
            val file = chapterDir.subFolder(PROCESSED).writeFile(name, drawn)
            PostProcessStore.savePage(context, request.chapterId, record.originalHash, file.uri.toString())
        } else {
            UniFile.fromUri(context, record.cleanUri)?.delete()
            chapterDir.writeFile(name, drawn)
            PostProcessStore.savePage(context, request.chapterId, PageColorizer.contentHash(drawn), "")
        }
    }

    /** Deletes the processed files and the translation files of a chapter and forgets them. Empty folders go too. */
    private fun clearResults(context: Context, chapterId: Long) {
        val folders = mutableSetOf<UniFile>()
        val uris = PostProcessStore.chapterUris(context, chapterId) + PostProcessStore.chapterTranslationUris(context, chapterId)
        uris.forEach { uri ->
            val file = UniFile.fromUri(context, Uri.parse(uri)) ?: return@forEach
            file.parentFile?.let { folders.add(it) }
            file.delete()
        }
        PostProcessStore.deleteChapter(context, chapterId)
        folders.forEach { if (it.listFiles()?.isEmpty() == true) it.delete() }
    }

    /** The folder with this name inside the chapter folder, created when it is missing. */
    private fun UniFile.subFolder(name: String): UniFile =
        findFile(name)?.takeIf { it.isDirectory } ?: createDirectory(name) ?: error("Cannot create the \"$name\" folder")

    /** Writes a file, replacing one with the same name (a file is never overwritten in place). */
    private fun UniFile.writeFile(name: String, bytes: ByteArray): UniFile {
        findFile(name)?.delete()
        val file = createFile(name) ?: error("Cannot create $name")
        file.openOutputStream().use { it.write(bytes) }
        return file
    }

    private fun extensionOf(bytes: ByteArray): String =
        if (bytes.size > 4 && bytes[0] == 0x89.toByte() && bytes[1] == 'P'.code.toByte()) ".png" else ".jpg"
}
