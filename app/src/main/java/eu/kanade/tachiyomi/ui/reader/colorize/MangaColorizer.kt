package eu.kanade.tachiyomi.ui.reader.colorize

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat

/**
 * Runs Manga Colorization V2 ONNX (`manga-colorize-fp16.onnx`) with a [1, 5, H, W] input tensor:
 *  - Channel 0: grayscale in [0, 1]
 *  - Channels 1..3: color hints in [-1, 1] (0 when no hints)
 *  - Channel 4: hint mask (0 when no hints)
 *
 * Not thread safe: call it from one coroutine at a time.
 */
class MangaColorizer(
    modelFile: File,
    private val acceleration: Acceleration = Acceleration.CPU,
    private val threads: Int = 0,
    @Suppress("UNUSED_PARAMETER") private val guide: SamFeatures? = null,
) : Closeable {

    class Chroma(
        val width: Int,
        val height: Int,
        val a: FloatArray,
        val b: FloatArray,
    ) {
        val planeSize: Int get() = width * height
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        check(OpenCVLoader.initLocal()) { "OpenCV native library failed to load" }
        val options = OrtSession.SessionOptions().apply {
            if (threads > 0) {
                setIntraOpNumThreads(threads)
            }
            when (acceleration) {
                Acceleration.CPU -> Unit
                else -> {
                    runCatching { addNnapi() }
                }
            }
        }
        session = env.createSession(modelFile.absolutePath, options)
        if (session.inputNames.isEmpty() || session.outputNames.isEmpty()) {
            session.close()
            error("Invalid ONNX colorizer model: missing inputs or outputs")
        }
    }

    fun predict(input: Bitmap): Chroma = Mats().use { m ->
        val (targetW, targetH) = computeModelSize(input.width, input.height)
        val plane = targetW * targetH

        val rgba = m.mat()
        toMat(input, rgba)
        val gray = m.mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        val small = m.mat()
        Imgproc.resize(gray, small, Size(targetW.toDouble(), targetH.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        // Channel 0 expects grayscale in [0, 1].
        small.convertTo(small, CvType.CV_32F, 1.0 / 255.0, 0.0)
        val grayPlane = FloatArray(plane)
        small.get(0, 0, grayPlane)

        // Build [1, 5, targetH, targetW] tensor: [gray(1), hints(3), mask(1)].
        val input5Ch = FloatArray(5 * plane)
        System.arraycopy(grayPlane, 0, input5Ch, 0, plane)

        val rgbPlanes = runModel(input5Ch, targetW, targetH)
        val planes = List(3) { c ->
            m.mat(targetH, targetW, CvType.CV_32F).also {
                it.put(0, 0, rgbPlanes.copyOfRange(c * plane, (c + 1) * plane))
            }
        }
        val merged = m.mat()
        Core.merge(planes, merged)
        // Model output is typically in [-1, 1]; map to [0, 1] and clamp.
        merged.convertTo(merged, CvType.CV_32F, 0.5, 0.5)
        Core.max(merged, Scalar(0.0, 0.0, 0.0), merged)
        Core.min(merged, Scalar(1.0, 1.0, 1.0), merged)

        val lab = m.mat()
        Imgproc.cvtColor(merged, lab, Imgproc.COLOR_RGB2Lab)
        val aMat = m.mat()
        val bMat = m.mat()
        Core.extractChannel(lab, aMat, 1)
        Core.extractChannel(lab, bMat, 2)
        Chroma(targetW, targetH, aMat.toFloatArray(plane), bMat.toFloatArray(plane))
    }

    /**
     * Builds the colored page: luminance from [input], chroma from [chroma] scaled by [intensity]
     * (0 = no color, 1 = model color, up to 1.2). Alpha is preserved.
     */
    fun render(input: Bitmap, chroma: Chroma, intensity: Float = 1f): Bitmap {
        require(intensity in 0f..1.2f) { "intensity must be within 0..1.2" }
        val width = input.width
        val height = input.height
        return Mats().use { m ->
            val rgba = m.mat()
            toMat(input, rgba)
            val rgb = m.mat()
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            // 8-bit Lab: L is 0..255, a and b are offset by 128.
            val lab = m.mat()
            Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
            val luminance = m.mat()
            Core.extractChannel(lab, luminance, 0)

            val a = scaledChroma(m, chroma.a, chroma.width, chroma.height, intensity, width, height)
            val b = scaledChroma(m, chroma.b, chroma.width, chroma.height, intensity, width, height)
            Core.merge(listOf(luminance, a, b), lab)
            Imgproc.cvtColor(lab, rgb, Imgproc.COLOR_Lab2RGB)

            val out = m.mat()
            Imgproc.cvtColor(rgb, out, Imgproc.COLOR_RGB2RGBA)
            val alpha = m.mat()
            Core.extractChannel(rgba, alpha, 3)
            Core.insertChannel(alpha, out, 3)

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(out, bitmap)
            bitmap
        }
    }

    override fun close() {
        session.close()
    }

    private fun runModel(input5Ch: FloatArray, width: Int, height: Int): FloatArray {
        val inputName = session.inputNames.first()
        val outputName = session.outputNames.first()
        val plane = width * height
        try {
            OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(input5Ch),
                longArrayOf(1, 5, height.toLong(), width.toLong()),
            ).use { inputTensor ->
                session.run(mapOf(inputName to inputTensor)).use { result ->
                    val output = result.getOutput(0) as OnnxTensor
                    val shape = output.info.shape
                    if (shape.size != 4 || shape[1] != 3L) {
                        error("Unexpected colorizer output shape: ${shape.joinToString()}")
                    }
                    val outH = shape[2].toInt()
                    val outW = shape[3].toInt()
                    val outPlane = outH * outW
                    val raw = FloatArray(3 * outPlane)
                    output.floatBuffer.get(raw)

                    if (outW == width && outH == height && raw.size == 3 * plane) {
                        return raw
                    }

                    // Resize output planes back to (height, width) if model output dimensions differ.
                    Mats().use { m ->
                        val resizedPlanes = FloatArray(3 * plane)
                        for (c in 0 until 3) {
                            val srcMat = m.mat(outH, outW, CvType.CV_32F).apply {
                                put(0, 0, raw.copyOfRange(c * outPlane, (c + 1) * outPlane))
                            }
                            val dstMat = m.mat()
                            Imgproc.resize(
                                srcMat,
                                dstMat,
                                Size(width.toDouble(), height.toDouble()),
                                0.0,
                                0.0,
                                Imgproc.INTER_LINEAR,
                            )
                            dstMat.get(0, 0, resizedPlanes).also {
                                // Copy channel c into its offset
                            }
                            val chArr = FloatArray(plane)
                            dstMat.get(0, 0, chArr)
                            System.arraycopy(chArr, 0, resizedPlanes, c * plane, plane)
                        }
                        return resizedPlanes
                    }
                }
            }
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e) { "MangaColorizer: inference failed" }
            throw e
        }
    }

    private fun scaledChroma(
        m: Mats,
        values: FloatArray,
        srcW: Int,
        srcH: Int,
        intensity: Float,
        dstW: Int,
        dstH: Int,
    ): Mat {
        val small = m.mat(srcH, srcW, CvType.CV_32F)
        small.put(0, 0, values)
        val small8 = m.mat()
        small.convertTo(small8, CvType.CV_8U, intensity.toDouble(), 128.0)
        val full = m.mat()
        Imgproc.resize(small8, full, Size(dstW.toDouble(), dstH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        return full
    }

    private fun toMat(bitmap: Bitmap, destination: Mat) {
        val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) {
            bitmap
        } else {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
        try {
            Utils.bitmapToMat(argb, destination)
        } finally {
            if (argb !== bitmap) argb.recycle()
        }
    }

    private fun Mat.toFloatArray(size: Int): FloatArray = FloatArray(size).also { get(0, 0, it) }

    private class Mats : Closeable {
        private val owned = ArrayList<Mat>()

        fun mat(): Mat = Mat().also { owned.add(it) }

        fun mat(rows: Int, cols: Int, type: Int): Mat = Mat(rows, cols, type).also { owned.add(it) }

        override fun close() {
            owned.forEach { it.release() }
        }
    }

    private companion object {
        const val MAX_SIDE = 576
        const val MULTIPLE_OF = 32

        fun computeModelSize(width: Int, height: Int): Pair<Int, Int> {
            if (width <= 0 || height <= 0) return MAX_SIDE to MAX_SIDE
            val scale = minOf(1.0, MAX_SIDE.toDouble() / maxOf(width, height))
            val w = ((width * scale).toInt() / MULTIPLE_OF).coerceAtLeast(1) * MULTIPLE_OF
            val h = ((height * scale).toInt() / MULTIPLE_OF).coerceAtLeast(1) * MULTIPLE_OF
            return w to h
        }
    }
}
