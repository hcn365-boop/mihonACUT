package eu.kanade.translation.translator

import eu.kanade.tachiyomi.network.await
import eu.kanade.translation.model.PageTranslation
import eu.kanade.translation.model.TranslationBlock
import eu.kanade.translation.recognizer.TextRecognizerLanguage
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import tachiyomi.core.common.util.system.logcat
import java.io.IOException
import java.net.URLEncoder
import java.util.IdentityHashMap
import java.util.concurrent.TimeUnit

@Suppress("PropertyName", "MaxLineLength", "FunctionName")
class GoogleTranslator(
    override val fromLang: TextRecognizerLanguage,
    override val toLang: TextTranslatorLanguage,
) : TextTranslator {
    private val client1 = "gtx"
    private val okHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build()
    }
    private val urlPattern = Regex("(?i)(https?://\\S+|www\\.\\S+|\\S+\\.(com|net|org|io|me|cc|tv|info))")
    private val symbolsFilterPattern = Regex("""[_~•°^@#$&|\\-]""")
    private val SAFE_SEPARATOR = " . "

    // يكتشف الكلمات اللاتينية/الإنجليزية المتبقية في النص المترجم
    private val untranslatedWordPattern = Regex("""[a-zA-Z]{2,}""")

    // لغات لا تستخدم الأحرف اللاتينية — نُفحّص فيها الكلمات غير المترجمة
    private val nonLatinLanguages = setOf(
        TextTranslatorLanguage.ARABIC,
        TextTranslatorLanguage.JAPANESE,
        TextTranslatorLanguage.CHINESESIM,
        TextTranslatorLanguage.CHINESETRAD,
        TextTranslatorLanguage.KOREAN,
        TextTranslatorLanguage.RUSSIAN,
        TextTranslatorLanguage.GREEK_MODERN,
        TextTranslatorLanguage.HEBREW,
        TextTranslatorLanguage.HINDI,
        TextTranslatorLanguage.THAI,
        TextTranslatorLanguage.PERSIAN,
        TextTranslatorLanguage.URDU,
        TextTranslatorLanguage.BENGALI,
        TextTranslatorLanguage.TAMIL,
        TextTranslatorLanguage.TELUGU,
        TextTranslatorLanguage.MALAYALAM,
        TextTranslatorLanguage.KANNADA,
        TextTranslatorLanguage.GUJARATI,
        TextTranslatorLanguage.PANJABI_PUNJABI,
        TextTranslatorLanguage.MARATHI,
        TextTranslatorLanguage.ARMENIAN,
        TextTranslatorLanguage.GEORGIAN,
        TextTranslatorLanguage.UKRAINIAN,
        TextTranslatorLanguage.BELARUSIAN,
        TextTranslatorLanguage.BULGARIAN,
        TextTranslatorLanguage.SERBIAN,
        TextTranslatorLanguage.MACEDONIAN,
        TextTranslatorLanguage.MONGOLIAN,
        TextTranslatorLanguage.KAZAKH,
        TextTranslatorLanguage.KIRGHIZ_KYRGYZ,
        TextTranslatorLanguage.TAJIK,
        TextTranslatorLanguage.TIBETAN,
        TextTranslatorLanguage.LAO,
        TextTranslatorLanguage.CENTRAL_KHMER,
        TextTranslatorLanguage.BURMESE,
        TextTranslatorLanguage.SINHALA_SINHALESE,
        TextTranslatorLanguage.AMHARIC,
        TextTranslatorLanguage.KURDISH,
        TextTranslatorLanguage.PUSHTO_PASHTO,
        TextTranslatorLanguage.UIGHUR_UYGHUR,
        TextTranslatorLanguage.SINDHI,
        TextTranslatorLanguage.YIDDISH,
        TextTranslatorLanguage.DIVEHI_DHIVEHI_MALDIVIAN,
        TextTranslatorLanguage.NEPALI,
        TextTranslatorLanguage.DZONGKHA,
    )

    // هل لغة الترجمة الحالية لا تستخدم لاتيني؟
    private val shouldFixUntranslated = nonLatinLanguages.contains(toLang)

    override suspend fun translate(pages: MutableMap<String, PageTranslation>) {
        val rtlLanguages = setOf(
            TextTranslatorLanguage.ARABIC,
            TextTranslatorLanguage.PERSIAN,
            TextTranslatorLanguage.HEBREW,
            TextTranslatorLanguage.URDU,
            TextTranslatorLanguage.PUSHTO_PASHTO,
            TextTranslatorLanguage.SINDHI,
            TextTranslatorLanguage.UIGHUR_UYGHUR,
            TextTranslatorLanguage.YIDDISH,
            TextTranslatorLanguage.KURDISH,
        )
        val rtlMarker = if (rtlLanguages.contains(toLang)) "\u200F" else ""

        // 1. The blocks that can be translated.
        val valid = mutableListOf<TranslationBlock>()
        pages.forEach { (_, page) ->
            page.blocks.forEach { block ->
                val isAcceptedAngle = (block.angle >= -15.0f && block.angle <= 15.0f) ||
                    (block.angle >= 75.0f && block.angle <= 105.0f) ||
                    (block.angle <= -75.0f && block.angle >= -105.0f)
                val isUrl = urlPattern.containsMatchIn(block.text)

                if (isAcceptedAngle && !isUrl) {
                    valid.add(block)
                } else {
                    block.translation = ""
                }
            }
        }
        if (valid.isEmpty()) return

        // 2. A phrase translated before (a sound effect, "yes", a name) needs no request.
        val translations = IdentityHashMap<TranslationBlock, String>()
        val pending = mutableListOf<TranslationBlock>()
        for (block in valid) {
            val known = cachedPhrase(cleanSymbols(block.text))
            if (known != null) translations[block] = known else pending.add(block)
        }

        // 3. Batches of at most 1700 characters (the text travels inside the address).
        val batches = mutableListOf<MutableList<TranslationBlock>>()
        var current = mutableListOf<TranslationBlock>()
        var length = 0
        for (block in pending) {
            val size = cleanSymbols(block.text).length + SAFE_SEPARATOR.length
            if (length + size > MAX_BATCH_CHARS && current.isNotEmpty()) {
                batches.add(current)
                current = mutableListOf()
                length = 0
            }
            current.add(block)
            length += size + 1
        }
        if (current.isNotEmpty()) batches.add(current)

        // 4. One request after another. The policy spaces them out, so a long chapter cannot get us blocked.
        val fresh = IdentityHashMap<TranslationBlock, String>()
        var lastError: Exception? = null
        for (batch in batches) {
            try {
                val lines = translateBatch(batch)
                batch.forEachIndexed { index, block ->
                    val line = lines.getOrNull(index)
                    if (!line.isNullOrBlank()) fresh[block] = line
                }
            } catch (e: TranslationRateLimitedException) {
                // Stop at once: more requests would only make the block longer.
                throw e
            } catch (e: IOException) {
                lastError = e
                logcat { "Translation failed: ${e.message}" }
            } catch (e: IllegalStateException) {
                lastError = e
                logcat { "Translation failed: ${e.message}" }
            }
        }
        // Nothing was translated and something failed: tell the caller instead of leaving empty bubbles.
        if (fresh.isEmpty() && translations.isEmpty() && lastError != null) throw lastError

        // 5. Words left in the original alphabet are translated in ONE request for all the batches.
        val blocks = fresh.keys.toList()
        val fixed = fixUntranslatedWordsBatch(blocks.map { fresh.getValue(it) }, toLang.code)
        blocks.forEachIndexed { index, block ->
            translations[block] = fixed[index]
            rememberPhrase(cleanSymbols(block.text), fixed[index])
        }

        // 6. Every block gets its translation.
        for (block in valid) {
            translations[block]?.let { block.translation = rtlMarker + it }
        }
    }

    /**
     * Translates the blocks of one batch in one request and returns one text per block, in order.
     * If the service merges or splits lines, the batch is cut in two and each half is translated apart, so every
     * bubble gets its own text and never the text of its neighbour.
     */
    private suspend fun translateBatch(batch: List<TranslationBlock>): List<String> {
        val merged = batch.joinToString("\n") { cleanSymbols(it.text) + SAFE_SEPARATOR }
        val translated = translateText(toLang.code, merged)
        val lines = translated.split("\n").map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.size == batch.size) return lines.map { tidy(it) }
        if (batch.size == 1) return listOf(tidy(lines.joinToString(" ")))
        val half = batch.size / 2
        return translateBatch(batch.subList(0, half)) + translateBatch(batch.subList(half, batch.size))
    }

    private fun tidy(text: String): String = text.removeSuffix(".").removeSuffix(" .").trim()

    /**
     * Collects the words that stayed in the original alphabet in every text and translates them
     * in a single request. A failure here is not fatal: the texts stay as they are.
     */
    private suspend fun fixUntranslatedWordsBatch(translatedTexts: List<String>, langCode: String): List<String> {
        if (!shouldFixUntranslated) return translatedTexts

        val allUntranslatedWords = translatedTexts.flatMap { text ->
            untranslatedWordPattern.findAll(text).map { it.value }.toList()
        }.distinct()
        if (allUntranslatedWords.isEmpty()) return translatedTexts

        val translatedWordsMerged = try {
            translateText(langCode, allUntranslatedWords.joinToString("\n"))
        } catch (e: TranslationRateLimitedException) {
            logcat { "Skipped fixing the remaining words: ${e.message}" }
            ""
        } catch (e: IOException) {
            logcat { "Failed to translate missing words batch: ${e.message}" }
            ""
        } catch (e: IllegalStateException) {
            logcat { "Failed to translate missing words batch: ${e.message}" }
            ""
        }

        val translationMap = mutableMapOf<String, String>()
        if (translatedWordsMerged.isNotBlank()) {
            val translatedWordsList = translatedWordsMerged.split("\n").map { it.trim() }
            allUntranslatedWords.forEachIndexed { index, word ->
                if (index < translatedWordsList.size) {
                    val single = tidy(translatedWordsList[index])
                    if (single.isNotBlank() && !untranslatedWordPattern.containsMatchIn(single)) {
                        translationMap[word] = single
                    }
                }
            }
        }

        return translatedTexts.map { originalText ->
            var fixedText = originalText
            translationMap.forEach { (word, translatedWord) ->
                // Word boundaries keep a word from being replaced inside another word.
                fixedText = fixedText.replace(Regex("""\b$word\b"""), translatedWord)
            }
            fixedText
        }
    }

    private fun cleanSymbols(rawText: String): String {
        return rawText.replace("\n", " ")
            .replace(symbolsFilterPattern, " ")
            .replace(Regex("""\s+"""), " ")
            .trim()
    }

    /** One request, under the shared policy. Every answer is checked before it is read. */
    private suspend fun translateText(lang: String, text: String): String = policy.execute {
        val request: Request = Request.Builder()
            .url(getTranslateUrl(lang, text))
            .header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            .build()

        okHttpClient.newCall(request).await().use { response ->
            RequestPolicy.checkStatus(response.code, response.header("Retry-After"))
            val body = response.body?.string().orEmpty()
            // A web page instead of the answer means the service is blocking us.
            if (!body.trimStart().startsWith("[")) {
                throw TranslationRateLimitedException("Google answered with a web page instead of a translation")
            }
            try {
                val sentences = JSONArray(body).getJSONArray(0)
                val result = StringBuilder()
                for (i in 0 until sentences.length()) {
                    val sentence = sentences.getJSONArray(i)
                    if (!sentence.isNull(0)) result.append(sentence.getString(0))
                }
                result.toString()
            } catch (e: Exception) {
                throw IllegalStateException("Cannot read the answer of Google: ${e.message}", e)
            }
        }
    }

    private fun getTranslateUrl(lang: String, text: String): String {
        val encode: String = URLEncoder.encode(text, "utf-8")
        val tk = calculateToken(text)

        return "https://translate.google.com/translate_a/single" +
            "?client=$client1" +
            "&sl=auto" +
            "&tl=$lang" +
            "&tk=$tk" +
            "&q=$encode" +
            "&dt=t" +
            "&ie=UTF-8" +
            "&oe=UTF-8"
    }

    private fun calculateToken(str: String): String {
        var j: Long = 406644
        val list = mutableListOf<Int>()
        var i = 0
        while (i < str.length) {
            val codePoint = str.codePointAt(i)
            if (codePoint < 128) {
                list.add(codePoint)
            } else if (codePoint < 2048) {
                list.add((codePoint shr 6) or 192)
                list.add((codePoint and 63) or 128)
            } else {
                list.add((codePoint shr 12) or 224)
                list.add(((codePoint shr 6) and 63) or 128)
                list.add((codePoint and 63) or 128)
            }
            i += if (Character.isSupplementaryCodePoint(codePoint)) 2 else 1
        }
        for (num in list) j = RL(j + num.toLong(), "+-a^+6")
        var rL = RL(j, "+-3^+b+-f") xor 3293161072L
        if (rL < 0) rL = (rL and 2147483647L) + 2147483648L
        val res = (rL % 1000000L)
        return "$res.${406644L xor res}"
    }

    private fun RL(j: Long, str: String): Long {
        var res = j
        var i = 0
        while (i < str.length - 2) {
            val shift = if (str[i + 2] in 'a'..'z') str[i + 2].code - 'W'.code else str[i + 2].digitToInt()
            val shiftValue = if (str[i + 1] == '+') res ushr shift else res shl shift
            res = if (str[i] == '+') (res + shiftValue) and 4294967295L else res xor shiftValue
            i += 3
        }
        return res
    }

    override fun close() {}

    companion object {
        /** One policy for all the Google requests of the app: the limits belong to the service. */
        private val policy = RequestPolicy(name = "Google Translate", minIntervalMs = 800, maxConcurrent = 1)

        private const val MAX_BATCH_CHARS = 1700
        private const val MAX_PHRASES = 1000

        /** Phrases already translated, per target language, to avoid sending the same text again. */
        private val phrases = object : LinkedHashMap<String, String>(256, 0.75f, true) {
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, String>): Boolean =
                size > MAX_PHRASES
        }
    }

    private fun cachedPhrase(cleaned: String): String? =
        if (cleaned.isEmpty()) null else synchronized(phrases) { phrases["${toLang.code}|$cleaned"] }

    private fun rememberPhrase(cleaned: String, translation: String) {
        if (cleaned.isEmpty() || translation.isBlank()) return
        synchronized(phrases) { phrases["${toLang.code}|$cleaned"] = translation }
    }
}
