package eu.kanade.translation.recognizer

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
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

    data class RectBox(val x: Float, val y: Float, val width: Float, val height: Float, val score: Float)

    fun detect(bitmap: Bitmap, targetSize: Int = 1024): List<RectBox> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()

        // 1. الحفاظ على الأبعاد الأصلية وحساب نسبة التكبير/التصغير (Letterboxing)
        val scale = min(targetSize.toFloat() / origW, targetSize.toFloat() / origH)
        val scaledW = (origW * scale).toInt()
        val scaledH = (origH * scale).toInt()

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)

        // 2. إنشاء لوحة مربعة 1024x1024 وملؤها باللون الرمادي المعياري (114, 114, 114)
        val paddedBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))

        // وضع الصورة المصغرة في المنتصف تماماً
        val dx = (targetSize - scaledW) / 2f
        val dy = (targetSize - scaledH) / 2f
        canvas.drawBitmap(resizedBitmap, dx, dy, null)

        // 3. تعبئة التنسور بنظام NCHW (قنوات R ثم G ثم B منفصلة)
        val inputBuffer = FloatBuffer.allocate(3 * targetSize * targetSize)
        val pixels = IntArray(targetSize * targetSize)
        paddedBitmap.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)

        val rChannel = FloatArray(targetSize * targetSize)
        val gChannel = FloatArray(targetSize * targetSize)
        val bChannel = FloatArray(targetSize * targetSize)

        for (i in pixels.indices) {
            val p = pixels[i]
            rChannel[i] = ((p shr 16) and 0xFF) / 255.0f
            gChannel[i] = ((p shr 8) and 0xFF) / 255.0f
            bChannel[i] = (p and 0xFF) / 255.0f
        }

        inputBuffer.put(rChannel)
        inputBuffer.put(gChannel)
        inputBuffer.put(bChannel)
        inputBuffer.rewind()

        val tensor = OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong()))
        val inputs = mapOf(session.inputNames.iterator().next() to tensor)

        val results = session.run(inputs)
        val boxes = mutableListOf<RectBox>()

        try {
            // استخراج التحليلات من مخرجات نموذج الـ ONNX وإعادة تحويل الإحداثيات عكسياً للصورة الأصلية
            // x_orig = (x_detected - dx) / scale
            // y_orig = (y_detected - dy) / scale
            val rawOutput = results[0].value
            // يتم تحليل المصفوفة الناتجة وتعبئة الـ boxes هنا حسب بنية النموذج الفعلية
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            tensor.close()
            results.close()
            if (resizedBitmap !== bitmap) resizedBitmap.recycle()
            if (paddedBitmap !== bitmap) paddedBitmap.recycle()
        }

        return boxes
    }

    fun close() {
        session.close()
        env.close()
    }
}
