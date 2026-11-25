package eu.kanade.tachiyomi.ui.reader.colorize

import android.graphics.Bitmap
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.BorderTypes
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/** Making pages smaller using Lanczos2 (a = 2) via OpenCV for sharp manga line-art preservation. */
internal object PageResizer {

    /** Longest side allowed for a processed page. JPEG cannot store more than 65,535 pixels on a side. */
    const val MAX_SIDE = 30_000
    private const val LANCZOS_A = 2.0

    @Volatile
    private var openCvLoaded = false

    private fun ensureOpenCv() {
        if (openCvLoaded) return
        synchronized(this) {
            if (!openCvLoaded) {
                check(OpenCVLoader.initLocal()) { "OpenCV native library failed to load" }
                openCvLoaded = true
            }
        }
    }

    /**
     * How much an image of this size must be scaled so that it has at most [maxPixels] pixels and its
     * longest side is at most [maxSide]. 1.0 means it already fits.
     */
    fun fitRatio(width: Int, height: Int, maxPixels: Long, maxSide: Int): Double {
        var ratio = 1.0
        val pixels = width.toLong() * height
        if (pixels > maxPixels) ratio = min(ratio, sqrt(maxPixels.toDouble() / pixels))
        val side = max(width, height)
        if (side * ratio > maxSide) ratio = maxSide.toDouble() / side
        return ratio
    }

    /**
     * Shrinks the page so its short side is [maxShortSide] pixels, keeping the proportions.
     * The short side (not the long one) tells the real resolution of any page: a wide double page
     * and a tall strip both keep their detail.
     *
     * @return [bitmap] itself when it is not larger than the limit, otherwise a new smaller bitmap
     */
    fun shrink(bitmap: Bitmap, maxShortSide: Int): Bitmap {
        val shortSide = min(bitmap.width, bitmap.height)
        if (maxShortSide <= 0 || shortSide <= maxShortSide) return bitmap
        val ratio = maxShortSide.toDouble() / shortSide
        return resizeTo(bitmap, (bitmap.width * ratio).roundToInt(), (bitmap.height * ratio).roundToInt())
    }

    /**
     * Scales the bitmap down so it fits [maxPixels] and [maxSide]. [bitmap] is recycled when a new one is made.
     * @return [bitmap] itself when it already fits
     */
    fun fitTo(bitmap: Bitmap, maxPixels: Long, maxSide: Int = MAX_SIDE): Bitmap {
        val ratio = fitRatio(bitmap.width, bitmap.height, maxPixels, maxSide)
        if (ratio >= 1.0) return bitmap
        val smaller = resizeTo(bitmap, (bitmap.width * ratio).roundToInt(), (bitmap.height * ratio).roundToInt())
        bitmap.recycle()
        return smaller
    }

    /** A new bitmap of exactly [width] x [height] pixels. [bitmap] is not changed. */
    fun resizeTo(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        ensureOpenCv()
        val w = max(1, width)
        val h = max(1, height)
        if (w == bitmap.width && h == bitmap.height) {
            return bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
        }

        val source = Mat()
        val target = Mat()
        try {
            Utils.bitmapToMat(bitmap, source)
            resizeMatLanczos2(source, target, w, h)
            val output = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(target, output)
            return output
        } finally {
            source.release()
            target.release()
        }
    }

    /** Resizes any OpenCV [Mat] using separable Lanczos2 (a = 2) when downscaling, or Lanczos when upscaling. */
    internal fun resizeMatLanczos2(src: Mat, dst: Mat, dstW: Int, dstH: Int) {
        val w = max(1, dstW)
        val h = max(1, dstH)
        if (w == src.cols() && h == src.rows()) {
            src.copyTo(dst)
            return
        }
        if (w >= src.cols() && h >= src.rows()) {
            Imgproc.resize(src, dst, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LANCZOS4)
            return
        }

        val scaleX = w.toDouble() / src.cols()
        val scaleY = h.toDouble() / src.rows()
        val kernelX = build1DKernel(scaleX)
        val kernelY = build1DKernel(scaleY)
        val filtered = Mat()
        try {
            Imgproc.sepFilter2D(
                src,
                filtered,
                -1,
                kernelX,
                kernelY,
                Point(-1.0, -1.0),
                0.0,
                BorderTypes.BORDER_REPLICATE,
            )
            Imgproc.resize(filtered, dst, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        } finally {
            kernelX.release()
            kernelY.release()
            filtered.release()
        }
    }

    private fun lanczos2Kernel(x: Double): Double {
        val ax = abs(x)
        if (ax < 1e-8) return 1.0
        if (ax >= LANCZOS_A) return 0.0
        val pix = PI * ax
        val pixOverA = pix / LANCZOS_A
        return (sin(pix) / pix) * (sin(pixOverA) / pixOverA)
    }

    private fun build1DKernel(scale: Double): Mat {
        val effectiveScale = min(1.0, scale)
        val support = LANCZOS_A / effectiveScale
        val radius = max(1, ceil(support).toInt())
        val size = radius * 2 + 1
        val weights = FloatArray(size)
        var sum = 0.0
        for (i in -radius..radius) {
            val w = lanczos2Kernel(i * effectiveScale)
            weights[i + radius] = w.toFloat()
            sum += w
        }
        if (sum > 0.0) {
            for (i in weights.indices) {
                weights[i] = (weights[i] / sum).toFloat()
            }
        }
        val kernel = Mat(1, size, CvType.CV_32F)
        kernel.put(0, 0, weights)
        return kernel
    }
}
