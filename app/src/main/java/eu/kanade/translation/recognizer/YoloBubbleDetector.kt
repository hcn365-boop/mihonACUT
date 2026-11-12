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
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat

class YoloBubbleDetector(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    
    // إعادة استخدام الـ buffers لتجنب GC pressure مع 25 صفحة
    private var inputBuffer: FloatBuffer? = null
    private var pixels: IntArray? = null

    init {
        val inputStream = context.assets.open("yolo/yolo26s.onnx")
        val bytes = inputStream.readBytes()
        inputStream.close()
        session = env.createSession(bytes)
        logcat(LogPriority.ERROR) { "YOLO26 Model loaded (NMS-enabled)" }
    }

    data class RectBox(
        val x: Float, 
        val y: Float, 
        val width: Float, 
        val height: Float, 
        val score: Float
    )

    /**
     * الكشف مع NMS
     * @param bitmap صورة الصفحة
     * @param targetSize يجب أن يكون 1280 (دقة التدريب)
     * @param confThreshold عتبة الثقة (0.25 افتراضياً)
     * @param iouThreshold عتبة التداخل للدمج (0.45 افتراضياً)
     */
    fun detect(
        bitmap: Bitmap, 
        targetSize: Int = 1280,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f
    ): List<RectBox> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()

        // 1. Resize مع الحفاظ على النسبة (Letterbox)
        val scale = min(targetSize.toFloat() / origW, targetSize.toFloat() / origH)
        val scaledW = (origW * scale).toInt()
        val scaledH = (origH * scale).toInt()

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
        val paddedBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114)) // خلفية الرمادي المعتمدة في YOLO
        val dx = (targetSize - scaledW) / 2f
        val dy = (targetSize - scaledH) / 2f
        canvas.drawBitmap(resizedBitmap, dx, dy, null)

        // 2. تجهيز الإدخال (CHW format)
        val bufferSize = 3 * targetSize * targetSize
        if (inputBuffer?.capacity() != bufferSize) {
            inputBuffer = FloatBuffer.allocate(bufferSize)
            pixels = IntArray(targetSize * targetSize)
        }
        
        val buf = inputBuffer!!
        val pix = pixels!!
        buf.clear()
        paddedBitmap.getPixels(pix, 0, targetSize, 0, 0, targetSize, targetSize)

        // فصل القنوات (R, G, B)
        for (i in pix.indices) {
            val p = pix[i]
            buf.put(i, ((p shr 16) and 0xFF) / 255.0f) // R
            buf.put(i + targetSize * targetSize, ((p shr 8) and 0xFF) / 255.0f) // G
            buf.put(i + 2 * targetSize * targetSize, (p and 0xFF) / 255.0f) // B
        }
        buf.rewind()

        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong()))
        val inputs = mapOf(session.inputNames.iterator().next() to tensor)

        val candidates = mutableListOf<RectBox>()
        
        try {
            val results = session.run(inputs)
            val rawOutput = results[0].value as Array<*>
            val detections = rawOutput[0] as Array<FloatArray> // [300][6]
            
            // 3. استخراج المرشحين (Candidate Extraction)
            for (det in detections) {
                val conf = det[4]
                if (conf > confThreshold) {
                    val xc = det[0]
                    val yc = det[1]
                    val w = det[2]
                    val h = det[3]
                    
                    // تحويل من الإحداثيات المُقاسة إلى الأصلية
                    val x1 = (xc - w / 2 - dx) / scale
                    val y1 = (yc - h / 2 - dy) / scale
                    val origWBox = w / scale
                    val origHBox = h / scale
                    
                    candidates.add(
                        RectBox(
                            x = x1.coerceIn(0f, origW),
                            y = y1.coerceIn(0f, origH),
                            width = origWBox.coerceAtMost(origW - x1.coerceIn(0f, origW)),
                            height = origHBox.coerceAtMost(origH - y1.coerceIn(0f, origH)),
                            score = conf
                        )
                    )
                }
            }
            results.close()
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e) { "YOLO26 inference failed" }
        } finally {
            tensor.close()
            resizedBitmap.recycle()
            paddedBitmap.recycle()
        }

        // 4. دمج المخرجات (NMS - Non-Maximum Suppression)
        val finalBoxes = applyNMS(candidates, iouThreshold)
        
        logcat(LogPriority.ERROR) { 
            "YOLO26: ${candidates.size} candidates -> ${finalBoxes.size} bubbles after NMS" 
        }
        
        return finalBoxes
    }

    /**
     * خوارزمية NMS لإزالة الصناديق المكررة
     * تبقي على الأعلى ثقة وتزيل المتداخلين معها
     */
    private fun applyNMS(boxes: List<RectBox>, iouThreshold: Float): List<RectBox> {
        if (boxes.isEmpty()) return emptyList()
        
        // ترتيب تنازلي حسب الثقة
        val sorted = boxes.sortedByDescending { it.score }
        val selected = mutableListOf<RectBox>()
        val suppressed = BooleanArray(sorted.size)
        
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            
            val current = sorted[i]
            selected.add(current)
            
            // قمع كل من يتداخل مع الحالي بشكل كبير
            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                
                if (calculateIoU(current, sorted[j]) > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        
        return selected
    }

    /**
     * حساب Intersection over Union (IoU)
     */
    private fun calculateIoU(a: RectBox, b: RectBox): Float {
        val x1 = max(a.x, b.x)
        val y1 = max(a.y, b.y)
        val x2 = min(a.x + a.width, b.x + b.width)
        val y2 = min(a.y + a.height, b.y + b.height)
        
        val intersectionArea = max(0f, x2 - x1) * max(0f, y2 - y1)
        val unionArea = (a.width * a.height) + (b.width * b.height) - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    fun close() {
        session.close()
        env.close()
        inputBuffer = null
        pixels = null
    }
}
