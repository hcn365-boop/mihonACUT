package eu.kanade.tachiyomi.ui.reader.colorize

import android.content.Context
import android.net.Uri
import com.hippo.unifile.UniFile
import eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences
import kotlinx.coroutines.CoroutineScope
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
import tachiyomi.core.common.util.system.logcat
import java.util.concurrent.ConcurrentHashMap
import kotlin.coroutines.cancellation.CancellationException

/**
 * Processes downloaded chapters in advance, one chapter at a time, with the batch settings (the "Batch"
 * tab) and not the real-time ones. Each processed page is saved in a "processed" folder inside the
 * chapter's download folder, next to the original pages, which are never changed. The reader then uses
 * the processed page when it finds one for the original (see [PostProcessStore]).
 */
object PostProcessManager {

    /** One chapter to process: its pages are the downloaded image files, in reading order. */
    class Request(
        val chapterId: Long,
        val mangaId: Long,
        val mangaTitle: String,
        val sourceLang: String,
        val pageUris: List<Uri>,
    )

    private class Entry(val context: Context, val prefs: ReaderPreferences, val request: Request)

    private const val FOLDER = "processed"

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
            val done = PostProcessStore.processedChapters(context)
            _states.update { current -> current + done.filter { it !in current }.associateWith { ChapterProcessState.Processed } }
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

    fun cancel(chapterId: Long) {
        if (runningId == chapterId) {
            runningJob?.cancel()
        } else if (_states.value[chapterId] == ChapterProcessState.Queued) {
            cancelled.add(chapterId)
            setState(chapterId, ChapterProcessState.NotProcessed)
        }
    }

    /** Deletes the processed pages of a chapter. The original pages stay. */
    fun delete(context: Context, chapterId: Long) {
        scope.launch {
            clearResults(context.applicationContext, chapterId)
            setState(chapterId, ChapterProcessState.NotProcessed)
        }
    }

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
        val job = scope.launch {
            try {
                setState(id, ChapterProcessState.Processing(0f))
                process(entry)
                PostProcessStore.setChapter(context, id, PostProcessStore.STATE_PROCESSED)
                PostProcessStore.refresh(context)
                setState(id, ChapterProcessState.Processed)
            } catch (e: CancellationException) {
                clearResults(context, id)
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
        val pages = request.pageUris
        if (pages.isEmpty()) error("The chapter has no downloaded pages")

        val chapterDir = UniFile.fromUri(context, pages.first())?.parentFile
            ?: error("Cannot find the chapter's download folder")
        val folder = chapterDir.findFile(FOLDER)?.takeIf { it.isDirectory }
            ?: chapterDir.createDirectory(FOLDER)
            ?: error("Cannot create the \"$FOLDER\" folder")

        // Start clean: earlier results may come from other settings.
        clearResults(context, request.chapterId)

        val snapshot = AiConfig.postSnapshot(entry.prefs, request.mangaTitle, request.sourceLang)
        AiConfig.withSnapshot(snapshot) {
            pages.forEachIndexed { index, uri ->
                kotlin.coroutines.coroutineContext.ensureActive()
                val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                    ?: error("Cannot read the page $uri")
                val hash = PageColorizer.contentHash(bytes)

                val processed = PageColorizer.processToBytes(context, Buffer().write(bytes))
                if (processed == null) {
                    PostProcessStore.savePage(context, request.chapterId, hash, "")
                } else {
                    val original = UniFile.fromUri(context, uri)?.name ?: "page_$index"
                    val name = original.substringBeforeLast('.') + if (isPng(processed)) ".png" else ".jpg"
                    folder.findFile(name)?.delete()
                    val file = folder.createFile(name) ?: error("Cannot create $name")
                    file.openOutputStream().use { it.write(processed) }
                    PostProcessStore.savePage(context, request.chapterId, hash, file.uri.toString())
                }
                setState(request.chapterId, ChapterProcessState.Processing((index + 1f) / pages.size))
            }
        }
    }

    /** Deletes the processed files of a chapter and forgets them. The "processed" folder goes when empty. */
    private fun clearResults(context: Context, chapterId: Long) {
        val folders = mutableSetOf<UniFile>()
        PostProcessStore.chapterUris(context, chapterId).forEach { uri ->
            val file = UniFile.fromUri(context, Uri.parse(uri)) ?: return@forEach
            file.parentFile?.let { folders.add(it) }
            file.delete()
        }
        PostProcessStore.deleteChapter(context, chapterId)
        folders.forEach { if (it.listFiles()?.isEmpty() == true) it.delete() }
    }

    private fun isPng(bytes: ByteArray): Boolean =
        bytes.size > 4 && bytes[0] == 0x89.toByte() && bytes[1] == 'P'.code.toByte()
}
