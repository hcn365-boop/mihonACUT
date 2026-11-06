package eu.kanade.translation

import android.content.Context
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import eu.kanade.translation.recognizer.MlKitTextRecognizer
import eu.kanade.translation.recognizer.PaddleTextRecognizer
import eu.kanade.translation.recognizer.TextRecognizerLanguage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

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

    private val effectiveMode = when {
        mode == OcrMode.MLKIT -> mode
        PaddleSupport.supports(context, language) -> mode
        else -> OcrMode.MLKIT
    }

    // ✅ تمرير context و recognizerLanguage إلى PageTextExtractor
    val extractor = PageTextExtractor(
        mode = effectiveMode,
        isRtl = language == TextRecognizerLanguage.JAPANESE || language == TextRecognizerLanguage.KOREAN,
        mlKit = if (effectiveMode == OcrMode.PADDLE) null else this::runMlKit,
        paddle = if (effectiveMode == OcrMode.MLKIT) null else this::runPaddle,
        context = context,              // ✅ إضافة هذا
        recognizerLanguage = language   // ✅ إضافة هذا
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

    override fun close() {
        if (mlKitEngine.isInitialized()) mlKitEngine.value.close()
        if (paddleEngine.isInitialized()) paddleEngine.value.close()
    }
}
