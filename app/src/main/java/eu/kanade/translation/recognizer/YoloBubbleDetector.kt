package eu.kanade.translation.recognizer

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.min

class YoloBubbleDetector(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        // تأكد من تطابق اسم ملف الـ ONNX في مجلد assets/yolo/
        val inputStream = context.assets.open("yolo/yolo26s.onnx")
        val bytes = inputStream.readBytes()
        inputStream.close()
        session = env.createSession(bytes)
    }

    data class RectBox(val x: Float, val y: Float, val width: Float, val height: Float, val score: Float)

    fun detect(bitmap: Bitmap, targetSize: Int = 1024): List<RectBox> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()

        // 1. الحفاظ على الأبعاد الأصلية (Letterboxing) حسب مواصفات النموذج
        val scale = min(targetSize.toFloat() / origW, targetSize.toFloat() / origH)
        val scaledW = (origW * scale).toInt()
        val scaledH = (origH * scale).toInt()

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)

        val paddedBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114)) // اللون الرمادي المعياري لـ YOLO

        val dx = (targetSize - scaledW) / 2f
        val dy = (targetSize - scaledH) / 2f
        canvas.drawBitmap(resizedBitmap, dx, dy, null)

        // 2. تجهيز التنسور بنظام NCHW وتطبيع الألوان (0.0 إلى 1.0)
        val inputBuffer = FloatBuffer.allocate(3 * targetSize * targetSize)
        val pixels = IntArray(targetSize * targetSize)
        paddedBitmap.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)

        val rChannel = FloatArray(targetSize * targetSize)
        val gChannel = FloatArray(targetSize * targetSize)
        val bChannel = FloatArray(targetStateSize(targetSize)) // bChannel size

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
            val rawOutput = results[0].value
            // معالجة مخرجات YOLO26 / YOLOv8 القياسية [1, 5/8400, N]
            if (rawOutput is Array<*>) {
                val batch0 = rawOutput[0]
                if (batch0 is Array<*>) {
                    // إذا كان الإخراج بتنسيق [Attr, Anchors]
                    if (batch0.isNotEmpty() && batch0[0] is FloatArray) {
                        val channels = batch0 as Array<FloatArray>
                        if (channels.size >= 5) {
                            val numAnchors = channels[0].size
                            for (i in 0 until numAnchors) {
                                val conf = channels[4][i]
                                if (conf > 0.25f) { // نسبة الثقة أكبر من 25%
                                    val xc = channels[0][i]
                                    val yc = channels[1][i]
                                    val w = channels[2][i]
                                    val h = channels[3][i]

                                    val x1 = xc - w / 2f
                                    val y1 = yc - h / 2f

                                    // عكس تأثير الـ Letterboxing والحجم لاستعادة الإحداثيات الأصلية للصورة
                                    val origX = (x1 - dx) / scale
                                    val origY = (y1 - dy) / scale
                                    val origWBox = w / scale
                                    val origHBox = h / scale

                                    boxes.add(
                                        RectBox(
                                            x = origX.coerceIn(0f, origW),
                                            y = origY.coerceIn(0f, origH),
                                            width = origWBox.coerceAtMost(origW - origX),
                                            height = origHBox.coerceAtMost(origH - origY),
                                            score = conf
                                        )
                                    )
                                }
                            }
                        }
                    } 
                    // إذا كان الإخراج بتنسيق [Anchors, Attr] (عكسي)
                    else if (batch0.isNotEmpty() && batch0[0] is FloatArray) {
                        // يتم التعامل معه إذا تطلب الأمر بناءً على شكل المصفوفة
                    }
                } else if (batch0 is FloatArray) {
                    // بعض النماذج تصدر مصفوفة مسطحة 2د
                }
            }
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

    private fun targetStateSize(size: Int) = size * size

    fun close() {
        session.close()
        env.close()
    }
}
