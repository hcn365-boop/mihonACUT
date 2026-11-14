package eu.kanade.translation.translator

import eu.kanade.tachiyomi.network.await
import eu.kanade.translation.model.PageTranslation
import eu.kanade.translation.recognizer.TextRecognizerLanguage
import logcat.logcat
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class LMStudioTranslator(
    override val fromLang: TextRecognizerLanguage,
    override val toLang: TextTranslatorLanguage,
    private val apiUrl: String, // The server link: http://10.0.0.10:1234/v1/chat/completions
    // The name of the model loaded in LM Studio. The old fixed name stays as the default.
    private val modelName: String = "qwen2.5-7b-instruct",
) : TextTranslator {

    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS) // A 7B model on a graphics card needs time to answer.
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    override suspend fun translate(pages: MutableMap<String, PageTranslation>) {
        try {
            val mediaType = "application/json; charset=utf-8".toMediaType()
            var total = 0
            var failures = 0
            var lastError: Exception? = null

            for ((_, page) in pages) {
                for (block in page.blocks) {
                    val cleanBlockText = block.text.trim().replace("\n", " ")
                    if (cleanBlockText.isBlank()) continue
                    total++

                    val systemInstruction = "You are an expert manga translator. Translate the following text from ${fromLang.name} to ${toLang.label}. Provide ONLY the natural, fluid translated text, without any explanations, thinking process, or additional formatting."

                    val root = JSONObject()
                    root.put("model", modelName)
                    root.put(
                        "messages",
                        JSONArray()
                            .put(JSONObject().put("role", "system").put("content", systemInstruction))
                            .put(JSONObject().put("role", "user").put("content", cleanBlockText)),
                    )
                    root.put("stream", false)
                    root.put("temperature", 0.7)

                    val request: Request = Request.Builder()
                        .url(apiUrl)
                        .header("Content-Type", "application/json")
                        .post(root.toString().toRequestBody(mediaType))
                        .build()

                    try {
                        // One request at a time: the graphics card answers one text at a time anyway.
                        val translatedText = policy.execute {
                            okHttpClient.newCall(request).await().use { response ->
                                RequestPolicy.checkStatus(response.code, response.header("Retry-After"))
                                val choices = JSONObject(response.body?.string().orEmpty()).optJSONArray("choices")
                                if (choices == null || choices.length() == 0) {
                                    ""
                                } else {
                                    choices.getJSONObject(0).optJSONObject("message")
                                        ?.optString("content", "").orEmpty()
                                        .trim('"', '\n', ' ', '`')
                                }
                            }
                        }
                        if (translatedText.isNotEmpty() && translatedText != cleanBlockText) {
                            block.translation = translatedText
                        }
                    } catch (e: TranslationRateLimitedException) {
                        throw e
                    } catch (e: Exception) {
                        // One bubble failing must not stop the others; the page fails only if all of them do.
                        if (e is kotlinx.coroutines.CancellationException) throw e
                        failures++
                        lastError = e
                        logcat { "LM Studio request failed: ${e.message}" }
                    }
                }
            }
            if (total > 0 && failures == total) throw lastError ?: IllegalStateException("LM Studio did not answer")
        } catch (e: Exception) {
            logcat { "LM Studio Translator Error : ${e.stackTraceToString()}" }
            throw e
        }
    }

    override fun close() {}

    companion object {
        /** A server of your own has no limit; one request at a time keeps the card busy without piling up. */
        private val policy = RequestPolicy(
            name = "LM Studio",
            minIntervalMs = 0,
            jitterMs = 0,
            maxConcurrent = 1,
            maxAttempts = 2,
        )
    }
}
