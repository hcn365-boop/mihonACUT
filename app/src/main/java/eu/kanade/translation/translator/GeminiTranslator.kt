package eu.kanade.translation.translator

import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.BlockThreshold
import com.google.ai.client.generativeai.type.HarmCategory
import com.google.ai.client.generativeai.type.SafetySetting
import com.google.ai.client.generativeai.type.content
import com.google.ai.client.generativeai.type.generationConfig
import eu.kanade.translation.model.PageTranslation
import eu.kanade.translation.recognizer.TextRecognizerLanguage
import kotlinx.coroutines.CancellationException
import logcat.logcat
import org.json.JSONObject
import java.io.IOException

@Suppress
class GeminiTranslator(
    override val fromLang: TextRecognizerLanguage,
    override val toLang: TextTranslatorLanguage,
    apiKey: String,
    modelName: String,
    val maxOutputToken: Int,
    val temp: Float,
) : TextTranslator {

    private var model: GenerativeModel = GenerativeModel(
        modelName = modelName,
        apiKey = apiKey,
        generationConfig = generationConfig {
            topK = 30
            topP = 0.5f
            temperature = temp
            maxOutputTokens = maxOutputToken
            responseMimeType = "application/json"
        },
        safetySettings = listOf(
            SafetySetting(HarmCategory.HARASSMENT, BlockThreshold.NONE),
            SafetySetting(HarmCategory.HATE_SPEECH, BlockThreshold.NONE),
            SafetySetting(HarmCategory.SEXUALLY_EXPLICIT, BlockThreshold.NONE),
            SafetySetting(HarmCategory.DANGEROUS_CONTENT, BlockThreshold.NONE),
        ),
        systemInstruction = content {
            text(
                "You are a highly skilled AI tasked with translating text from scanned images of comics (manhwa, manga, manhua). " +
                    "Translate the following JSON object values from ${fromLang.name} to ${toLang.label}.\n" +
                    "CRITICAL RULES:\n" +
                    "1. Return ONLY valid JSON matching the input structure exactly.\n" +
                    "2. Remove watermarks (e.g. site links) by replacing them with 'RTMTH'.\n" +
                    "3. DO NOT wrap the output in ```json ... ``` markdown blocks. Output the raw JSON object directly.",
            )
        },
    )

    override suspend fun translate(pages: MutableMap<String, PageTranslation>) {
        try {
            // 1. The texts of every page, as one JSON object.
            val data = pages.mapValues { (_, v) -> v.blocks.map { b -> b.text } }
            val json = JSONObject(data)

            // 2. One request, under the shared policy: spaced out, tried again when the service is busy,
            // and paused when the quota is reached.
            val response = policy.execute {
                try {
                    model.generateContent(json.toString())
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    throw classify(e)
                }
            }
            val rawText = response.text
            if (rawText.isNullOrBlank()) {
                throw IllegalStateException("Gemini returned no text (the answer was empty or blocked)")
            }

            // 3. Markdown marks around the JSON would break the parsing.
            val cleanText = rawText.replace("```json", "").replace("```", "").trim()
            val resJson = JSONObject(cleanText)

            // 4. The translations go to the blocks.
            for ((k, v) in pages) {
                v.blocks.forEachIndexed { i, b ->
                    val res = resJson.optJSONArray(k)?.optString(i, "NULL")
                    b.translation = if (res == null || res == "NULL") b.text else res
                }
                // 5. Blocks the model marked as watermarks are removed.
                v.blocks = v.blocks.filterNot { it.translation.contains("RTMTH") }.toMutableList()
            }
        } catch (e: Exception) {
            logcat { "Gemini Translation Error : ${e.stackTraceToString()}" }
            throw e
        }
    }

    /** Tells the policy what kind of failure this is: a quota to wait for, a busy server, or a real error. */
    private fun classify(e: Exception): Exception {
        val kind = e::class.simpleName.orEmpty()
        val text = e.message.orEmpty()
        return when {
            kind.contains("Quota", ignoreCase = true) ||
                text.contains("429") ||
                text.contains("RESOURCE_EXHAUSTED", ignoreCase = true) ||
                text.contains("quota", ignoreCase = true) ->
                TranslationRateLimitedException("Gemini: quota or rate limit reached. $text")

            kind.contains("Timeout", ignoreCase = true) ||
                kind.contains("Server", ignoreCase = true) ||
                text.contains("503") ||
                text.contains("UNAVAILABLE", ignoreCase = true) ->
                IOException("Gemini: temporary error. $text", e)

            else -> e
        }
    }

    override fun close() {}

    companion object {
        /** The free plan allows about 15 requests a minute (check your plan): one every 4.5 seconds. */
        private val policy = RequestPolicy(name = "Gemini", minIntervalMs = 4500, maxConcurrent = 1)
    }
}
