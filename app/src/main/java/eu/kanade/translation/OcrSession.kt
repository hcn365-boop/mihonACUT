package eu.kanade.translation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import eu.kanade.translation.recognizer.HybridTextRecognizer
import eu.kanade.translation.recognizer.MlKitTextRecognizer
import eu.kanade.translation.recognizer.PaddleTextRecognizer
import eu.kanade.translation.recognizer.TextRecognizerLanguage
import java.io.Closeable
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class OcrSession(
    private val context: Context,
    private val language: TextRecognizerLanguage,
    mode: OcrMode,
) : Closeable {

    init {
        Log.e("OcrSession", ">>> CREATED mode=$mode language=$language <<<")
    }

    private val mlKitEngine = lazy { MlKitTextRecognizer(language) }
    private val paddleEngine = lazy {
        Log.e("OcrSession", ">>> Paddle engine LAZY-INITIALIZED for $language <<<")
        PaddleTextRecognizer(context, language)
    }
    private val hybridEngine = lazy {
        Log.e("OcrSession", ">>> Hybrid engine LAZY-INITIALIZED for $language <<<")
        HybridTextRecognizer(context, language)
    }

    private val effectiveMode = when {
        mode == OcrMode.MLKIT -> mode
        PaddleSupport.supports(context, language) -> mode
        else -> OcrMode.MLKIT
    }

    val extractor = PageTextExtractor(
        mode = effectiveMode,
        isRtl = language == TextRecognizerLanguage.JAPANESE || language == TextRecognizerLanguage.KOREAN,
        mlKit = if (effectiveMode == OcrMode.PADDLE) null else this::runMlKit,
        paddle = if (effectiveMode == OcrMode.MLKIT) null else this::runPaddle,
        context = context,
        recognizerLanguage = language,
    )

    private suspend fun runMlKit(image: InputImage): Text {
        Log.e("OcrSession", ">>> runMlKit CALLED <<<")
        return withContext(Dispatchers.Default) { mlKitEngine.value.process(image) }
    }

    private suspend fun runPaddle(image: InputImage): List<OcrBox> {
        Log.e("OcrSession", ">>> runPaddle CALLED, delegating to PaddleTextRecognizer <<<")
        return withContext(Dispatchers.Default) {
            paddleEngine.value.process(image).map {
                OcrBox(text = it.text, x = it.x, y = it.y, width = it.width, height = it.height)
            }
        }
    }

    suspend fun runHybrid(image: InputImage): List<HybridTextRecognizer.HybridResult> {
        return withContext(Dispatchers.Default) {
            hybridEngine.value.recognizeWithBoxes(image)
        }
    }

    suspend fun recognizeHybridCrop(
        bitmap: Bitmap,
        x: Float,
        y: Float,
        width: Float,
        height: Float,
        angle: Float,
        fallbackText: String,
    ): String {
        return withContext(Dispatchers.Default) {
            val startX = max(0, x.roundToInt())
            val startY = max(0, y.roundToInt())
            val endX = min(bitmap.width, (x + width).roundToInt())
            val endY = min(bitmap.height, (y + height).roundToInt())

            val cropW = endX - startX
            val cropH = endY - startY
            if (cropW < 4 || cropH < 4) return@withContext fallbackText

            var cropped: Bitmap? = null
            var rotated: Bitmap? = null
            try {
                cropped = Bitmap.createBitmap(bitmap, startX, startY, cropW, cropH)
                rotated = if (abs(angle) > 1f) {
                    val matrix = Matrix().apply {
                        postRotate(-angle, cropW / 2f, cropH / 2f)
                    }
                    Bitmap.createBitmap(cropped, 0, 0, cropW, cropH, matrix, true)
                } else {
                    cropped
                }

                val cropInput = InputImage.fromBitmap(rotated, 0)
                hybridEngine.value.recognize(cropInput)
            } catch (e: Exception) {
                Log.e("OcrSession", "Failed to recognize hybrid crop at ($x, $y)", e)
                fallbackText
            } finally {
                if (rotated != null && rotated !== cropped && !rotated.isRecycled) {
                    rotated.recycle()
                }
                if (cropped != null && !cropped.isRecycled) {
                    cropped.recycle()
                }
            }
        }
    }

    override fun close() {
        if (mlKitEngine.isInitialized()) mlKitEngine.value.close()
        if (paddleEngine.isInitialized()) paddleEngine.value.close()
        if (hybridEngine.isInitialized()) hybridEngine.value.close()
    }
}
