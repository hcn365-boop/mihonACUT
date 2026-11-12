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
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat

class YoloBubbleDetector(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val inputStream = context.assets.open("yolo/yolo26s.onnx")
        val bytes = inputStream.readBytes()
        inputStream.close()
        session = env.createSession(bytes)
        logcat(LogPriority.ERROR) { "YOLO Model loaded successfully from assets/yolo/yolo26s.onnx" }
    }

    data class RectBox(val x: Float, val y: Float, val width: Float, val height: Float, val score: Float)

    fun detect(bitmap: Bitmap, targetSize: Int = 1024): List<RectBox> {
        val origW = bitmap.width.toFloat()
        val origH = bitmap.height.toFloat()
        logcat(LogPriority.ERROR) { "YOLO Original image size: ${origW}x${origH}" }

        val scale = min(targetSize.toFloat() / origW, targetSize.toFloat() / origH)
        val scaledW = (origW * scale).toInt()
        val scaledH = (origH * scale).toInt()

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)

        val paddedBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))

        val dx = (targetSize - scaledW) / 2f
        val dy = (targetSize - scaledH) / 2f
        canvas.drawBitmap(resizedBitmap, dx, dy, null)

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

        val boxes = mutableListOf<RectBox>()
        try {
            val results = session.run(inputs)
            val rawOutput = results[0].value
            logcat(LogPriority.ERROR) { "YOLO Model executed. Output type: ${rawOutput?.javaClass?.name}" }

            if (rawOutput is Array<*>) {
                val batch0 = rawOutput[0]
                if (batch0 is Array<*>) {
                    if (batch0.isNotEmpty() && batch0[0] is FloatArray) {
                        val channels = batch0 as Array<FloatArray>
                        if (channels.size >= 5) {
                            val numAnchors = channels[0].size
                            var detectedCount = 0
                            for (i in 0 until numAnchors) {
                                val conf = channels[4][i]
                                if (conf > 0.20f) {
                                    detectedCount++
                                    val xc = channels[0][i]
                                    val yc = channels[1][i]
                                    val w = channels[2][i]
                                    val h = channels[3][i]

                                    val x1 = xc - w / 2f
                                    val y1 = yc - h / 2f

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
                            logcat(LogPriority.ERROR) { "YOLO Bubbles passed confidence (>0.2): $detectedCount" }
                        }
                    }
                }
            }
            results.close()
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e) { "YOLO Error during session run or parsing" }
        } finally {
            tensor.close()
            if (resizedBitmap !== bitmap) resizedBitmap.recycle()
            if (paddedBitmap !== bitmap) paddedBitmap.recycle()
        }

        logcat(LogPriority.ERROR) { "YOLO Final detected boxes count = ${boxes.size}" }
        return boxes
    }

    fun close() {
        session.close()
        env.close()
    }
}
