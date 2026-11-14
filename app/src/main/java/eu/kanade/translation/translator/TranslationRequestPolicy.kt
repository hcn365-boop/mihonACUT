package eu.kanade.translation.translator

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.sync.withPermit
import java.io.IOException
import kotlin.random.Random

/** The service says "too many requests", or blocks us. The engine pauses; the message tells the user why. */
class TranslationRateLimitedException(message: String, val retryAfterMs: Long = 0L) : IOException(message)

/**
 * Keeps the requests to one translation service polite, so it does not block us:
 * - at most [maxConcurrent] requests run at the same time;
 * - at least [minIntervalMs] (plus a random part) pass between the starts of two requests;
 * - a temporary failure (timeout, server error) is tried again after a growing wait, up to [maxAttempts];
 * - when the service refuses us (HTTP 429 or 403, or a web page instead of an answer) the engine pauses for
 *   30 seconds, then 1 minute, 2 minutes... up to 15 minutes, and does not send anything meanwhile.
 *
 * Make ONE policy per service (a shared `companion object` value), not one per translator object, because the
 * limits belong to the service and not to a page or a chapter.
 */
class RequestPolicy(
    private val name: String,
    private val minIntervalMs: Long,
    private val jitterMs: Long = minIntervalMs / 3,
    maxConcurrent: Int = 1,
    private val maxAttempts: Int = 3,
) {
    private val slots = Semaphore(maxConcurrent)
    private val spacing = Mutex()
    private var nextStart = 0L

    @Volatile
    private var blockedUntil = 0L

    @Volatile
    private var strikes = 0

    /** Runs [block] (one request) under the policy. Throws the last error when it keeps failing. */
    suspend fun <T> execute(block: suspend () -> T): T {
        var attempt = 0
        while (true) {
            waitForCooldown()
            val outcome = slots.withPermit {
                waitForTurn()
                runCatching { block() }
            }
            val failure = outcome.exceptionOrNull()
            if (failure == null) {
                strikes = 0
                return outcome.getOrThrow()
            }
            if (failure is CancellationException) throw failure
            attempt++
            when (failure) {
                is TranslationRateLimitedException -> {
                    startCooldown(failure.retryAfterMs)
                    if (attempt >= maxAttempts) throw failure
                }
                is IOException -> {
                    if (attempt >= maxAttempts) throw failure
                    delay(backoff(attempt))
                }
                // Anything else (a refused request, a broken answer) will not improve by trying again.
                else -> throw failure
            }
        }
    }

    private suspend fun waitForTurn() {
        val wait = spacing.withLock {
            val now = System.currentTimeMillis()
            val start = maxOf(now, nextStart)
            nextStart = start + minIntervalMs + Random.nextLong(0, jitterMs + 1)
            start - now
        }
        if (wait > 0) delay(wait)
    }

    private suspend fun waitForCooldown() {
        val remaining = blockedUntil - System.currentTimeMillis()
        if (remaining <= 0) return
        // A short pause is waited out. A long one is reported at once instead of freezing the page.
        if (remaining > MAX_WAIT_MS) {
            throw TranslationRateLimitedException(
                "$name refused the requests; paused for ${remaining / 1000} more seconds",
                remaining,
            )
        }
        delay(remaining)
    }

    private fun startCooldown(retryAfterMs: Long) {
        strikes++
        val growing = (30_000L shl (strikes - 1).coerceAtMost(5)).coerceAtMost(MAX_COOLDOWN_MS)
        blockedUntil = maxOf(blockedUntil, System.currentTimeMillis() + maxOf(retryAfterMs, growing))
    }

    private fun backoff(attempt: Int): Long = (1000L shl (attempt - 1).coerceAtMost(4)) + Random.nextLong(0, 500)

    companion object {
        private const val MAX_WAIT_MS = 2 * 60_000L
        private const val MAX_COOLDOWN_MS = 15 * 60_000L

        /**
         * Call with the HTTP status of every answer. Does nothing for a success; otherwise throws the exception
         * that tells the policy what to do (pause, try again, or give up).
         */
        fun checkStatus(code: Int, retryAfterHeader: String?) {
            when {
                code in 200..299 -> Unit
                code == 429 || code == 403 -> throw TranslationRateLimitedException(
                    "HTTP $code: too many requests, or the service blocked this phone for a while",
                    (retryAfterHeader?.trim()?.toLongOrNull() ?: 0L) * 1000L,
                )
                code in 500..599 || code == 408 -> throw IOException("HTTP $code")
                else -> throw IllegalStateException("HTTP $code")
            }
        }
    }
}
