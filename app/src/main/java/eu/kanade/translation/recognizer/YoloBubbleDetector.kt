package eu.kanade.translation.recognizer

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class YoloBubbleDetector(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val inputStream = context.assets.open("yolo/yolo26s.onnx")
        val bytes = inputStream.readBytes()
        inputStream.close()
        session = env.createSession(bytes)
    }

    data.class RectBox(val x: Float, val y: Float, val width: Float, val height: Float, val score: Float)

    fun detect(bitmap: Bitmap, targetSize: Int = 1024): List<RectBox> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()

        // تغيير حجم الصورة لتناسب مدخلات النموذج (مثلاً 1024x1024)
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
        
        val inputBuffer = FloatBuffer.allocate(3 * targetSize * targetSize)
        for (y in 0 until targetSize) {
            for (x in 0 until targetSize) {
                val pixel = scaledBitmap.getPixel(x, y)
                inputBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)
                inputBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)
                inputBuffer.put((pixel and 0xFF) / 255.0f)
            }
        }
        inputBuffer.rewind()

        val tensor = OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong()))
        val inputs = mapOf(session.inputNames.iterator().next() to tensor)
        
        val results = session.run(inputs)
        val outputTensor = results[0].value as Array<*>
        
        // استخراج الصناديق (يختلف حسب بنية الإخراج لـ YOLOv8/26، وعادة يكون بتنسيق [x, y, w, h, conf])
        val boxes = mutableListOf<RectBox>()
        // تحليل مصفوفة النتائج وتحويلها لإحداثيات الصورة الأصلية (origW, origH)
        // ... (سيتم ضبطها حسب شكل إخراج النموذج الخاص بك)

        tensor.close()
        results.close()
        if (scaledBitmap !== bitmap) scaledBitmap.recycle()

        return boxes
    }

    fun close() {
        session.close()
        env.close()
    }
}
