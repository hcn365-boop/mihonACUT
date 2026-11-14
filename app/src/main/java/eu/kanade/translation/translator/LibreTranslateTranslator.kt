package eu.kanade.translation.translator

import eu.kanade.tachiyomi.network.await
import eu.kanade.translation.model.PageTranslation
import eu.kanade.translation.model.TranslationBlock
import eu.kanade.translation.recognizer.TextRecognizerLanguage
import logcat.logcat
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class LibreTranslateTranslator(
    override val fromLang: TextRecognizerLanguage,
    override val toLang: TextTranslatorLanguage,
    private val apiUrl: String,
) : TextTranslator {

    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    override suspend fun translate(pages: MutableMap<String, PageTranslation>) {
        try {
            // 1. Every block of every page, in order.
            val blocks = ArrayList<TranslationBlock>()
            for ((_, page) in pages) blocks.addAll(page.blocks)
            if (blocks.isEmpty()) return

            // 2. Requests of a sensible size: a server refuses one that is too big.
            for (chunk in chunks(blocks)) {
                val translated = requestChunk(chunk.map { it.text })
                chunk.forEachIndexed { index, block -> block.translation = translated.getString(index) }
            }
        } catch (e: Exception) {
            logcat { "LibreTranslate Error : ${e.stackTraceToString()}" }
            throw e
        }
    }

    /** Groups of at most [MAX_STRINGS] texts and [MAX_CHARS] characters. */
    private fun chunks(blocks: List<TranslationBlock>): List<List<TranslationBlock>> {
        val result = mutableListOf<List<TranslationBlock>>()
        var current = mutableListOf<TranslationBlock>()
        var chars = 0
        for (block in blocks) {
            if (current.isNotEmpty() && (current.size >= MAX_STRINGS || chars + block.text.length > MAX_CHARS)) {
                result.add(current)
                current = mutableListOf()
                chars = 0
            }
            current.add(block)
            chars += block.text.length
        }
        if (current.isNotEmpty()) result.add(current)
        return result
    }

    private suspend fun requestChunk(texts: List<String>): JSONArray {
        val jsonObject = JSONObject()
        jsonObject.put("q", JSONArray(texts))
        jsonObject.put("source", "auto")
        // The target language of the settings (LibreTranslate uses the same two-letter codes).
        jsonObject.put("target", toLang.code)
        jsonObject.put("format", "text")

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val request: Request = Request.Builder()
            .url(apiUrl)
            .header("Content-Type", "application/json")
            .post(jsonObject.toString().toRequestBody(mediaType))
            .build()

        return policy.execute {
            okHttpClient.newCall(request).await().use { response ->
                val text = response.body?.string().orEmpty()
                if (!response.isSuccessful) {
                    val reason = runCatching { JSONObject(text).getString("error") }.getOrNull()
                    try {
                        RequestPolicy.checkStatus(response.code, response.header("Retry-After"))
                    } catch (e: TranslationRateLimitedException) {
                        throw TranslationRateLimitedException("LibreTranslate: ${reason ?: e.message}", e.retryAfterMs)
                    } catch (e: IllegalStateException) {
                        throw IllegalStateException("LibreTranslate: ${reason ?: e.message}", e)
                    }
                }
                val array = JSONObject(text).optJSONArray("translatedText")
                if (array == null || array.length() != texts.size) {
                    throw IllegalStateException("LibreTranslate answered with ${array?.length() ?: 0} texts for ${texts.size}")
                }
                array
            }
        }
    }

    override fun close() {}

    companion object {
        private const val MAX_STRINGS = 40
        private const val MAX_CHARS = 3000

        /** A server of your own has no limit; a public one does, so the requests are spaced out a little. */
        private val policy = RequestPolicy(name = "LibreTranslate", minIntervalMs = 300, maxConcurrent = 1)
    }
}
