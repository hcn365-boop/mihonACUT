package eu.kanade.translation

import eu.kanade.translation.model.PageTranslation

/** Which filters run on the text of a page after OCR. */
class PageFilterOptions(
    /** Bubbles whose cleaned text contains one of these (already cleaned) words are removed. */
    val words: List<String>,
    /** Remove Latin bubbles with fewer than [latinMinLetters] different letters. */
    val latinEnabled: Boolean,
    val latinMinLetters: Int,
    /** Words the letter-count filter must keep (already cleaned), like "is", "or", "i". */
    val keepWords: List<String>,
    /** Also keep the common short English words of [PageFilter.COMMON_WORDS]. */
    val keepCommon: Boolean,
    /** Do not translate a page that is already in the target language, unless the user asks for it. */
    val skipSameLanguage: Boolean,
) {
    /** Every word the letter-count filter must keep. */
    val keepSet: Set<String> = if (keepCommon) keepWords.toSet() + PageFilter.COMMON_WORDS else keepWords.toSet()

    /** Changes whenever the result of the filters could change, so saved results are not reused wrongly. */
    val signature: String
        get() = "w=${words.joinToString(",")};l=$latinEnabled:$latinMinLetters;" +
            "k=${keepWords.joinToString(",")}:$keepCommon;s=$skipSameLanguage"

    companion object {
        /** Turns what the user typed (separated by commas, semicolons or new lines) into cleaned words. */
        fun parseWords(text: String): List<String> =
            text.split(',', ';', '\n').map { PageFilter.clean(it) }.filter { it.isNotEmpty() }.distinct()
    }
}

/**
 * Removes unwanted bubbles from the text of a page. A removed bubble is not translated and nothing is
 * drawn for it, so its original text stays on the page.
 *
 * Order: a bubble with a word from the delete list is always removed; otherwise a Latin bubble with too
 * few different letters is removed, unless every word in it is on the keep list.
 */
object PageFilter {

    /**
     * Very common short English words (apostrophes are dropped, so "it's" is "its"). With a high limit of
     * different letters, the letter-count filter would otherwise remove bubbles like "Is it?" or "The".
     */
    val COMMON_WORDS: Set<String> = setOf(
        "i", "a", "an", "am", "as", "at", "be", "by", "do", "go", "he", "hi", "if", "in", "is", "it", "me", "my",
        "no", "of", "ok", "on", "or", "so", "to", "up", "us", "we", "and", "all", "any", "are", "but", "can",
        "did", "for", "get", "got", "had", "has", "her", "him", "his", "how", "its", "let", "may", "not", "now",
        "off", "one", "our", "out", "own", "say", "see", "she", "the", "too", "two", "was", "who", "why", "yes",
        "yet", "you", "that", "this", "they", "them", "then", "what", "when", "will", "with", "your", "from",
        "have", "been", "were", "just", "like", "want", "know", "here", "there", "where", "which", "would",
        "could", "about", "think", "right", "going", "im", "ive", "ill", "id", "dont", "cant", "wont", "isnt",
        "arent", "wasnt", "didnt", "hes", "shes", "thats", "whats", "youre", "theyre", "lets",
    )

    /**
     * The text with digits, punctuation and symbols removed, in lower case. Apostrophes join the letters
     * around them ("don't" is "dont"); every other run of removed characters becomes one space, so
     * "Credit:SITE.com" becomes "credit site com".
     */
    fun clean(text: String): String {
        val out = StringBuilder()
        for (ch in text) {
            when {
                Character.isLetter(ch) -> out.append(ch.lowercaseChar())
                ch == '\'' || ch == '\u2019' -> Unit
                else -> if (out.isNotEmpty() && out.last() != ' ') out.append(' ')
            }
        }
        return out.toString().trim()
    }

    fun apply(page: PageTranslation, options: PageFilterOptions): PageTranslation {
        if (options.words.isEmpty() && !options.latinEnabled) return page
        val kept = page.blocks.filter { keep(it.text, options) }
        if (kept.size == page.blocks.size) return page
        val result = PageTranslation(imgWidth = page.imgWidth, imgHeight = page.imgHeight)
        result.blocks = kept.toMutableList()
        return result
    }

    private fun keep(text: String, options: PageFilterOptions): Boolean {
        val cleaned = clean(text)
        if (options.words.any { hasWord(cleaned, it) }) return false
        if (options.latinEnabled && isLatin(cleaned) && uniqueLetters(cleaned) < options.latinMinLetters) {
            // Short on different letters, but kept when it is only made of words the user wants to keep.
            return onlyKeptWords(cleaned, options.keepSet)
        }
        return true
    }

    /** True when the text has words and every one of them is in [keep]. */
    private fun onlyKeptWords(cleaned: String, keep: Set<String>): Boolean {
        val words = cleaned.split(' ').filter { it.isNotEmpty() }
        return words.isNotEmpty() && words.all { it in keep }
    }

    /** A whole word or phrase. Scripts written without spaces (Chinese, Japanese, Korean) match anywhere. */
    private fun hasWord(cleaned: String, word: String): Boolean {
        if (word.any { isUnspaced(it) }) return cleaned.replace(" ", "").contains(word.replace(" ", ""))
        return Regex("(^| )${Regex.escape(word)}( |$)").containsMatchIn(cleaned)
    }

    private fun isUnspaced(ch: Char): Boolean = when (Character.UnicodeScript.of(ch.code)) {
        Character.UnicodeScript.HAN,
        Character.UnicodeScript.HIRAGANA,
        Character.UnicodeScript.KATAKANA,
        Character.UnicodeScript.HANGUL,
        -> true
        else -> false
    }

    /** Has letters, and every letter is Latin. Text with no letters at all is not treated as Latin. */
    private fun isLatin(cleaned: String): Boolean {
        val letters = cleaned.filter { it.isLetter() }
        return letters.isNotEmpty() && letters.all { Character.UnicodeScript.of(it.code) == Character.UnicodeScript.LATIN }
    }

    private fun uniqueLetters(cleaned: String): Int = cleaned.filter { it.isLetter() }.toSet().size
}
