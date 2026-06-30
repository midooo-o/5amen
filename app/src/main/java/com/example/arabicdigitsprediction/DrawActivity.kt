package com.example.arabicdigitsprediction

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.os.Bundle
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Locale
import kotlin.math.roundToInt

class DrawActivity : AppCompatActivity() {

    companion object {
        private const val TARGET_SIZE = 28
        private const val DIGIT_SIZE = 20
        private const val WHITE_THRESHOLD = 20
    }

    private lateinit var drawingView: DigitDrawingView
    private lateinit var resultText: TextView
    private lateinit var previewImageView: ImageView
    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_draw)

        val drawingContainer = findViewById<FrameLayout>(R.id.drawingContainer)
        val predictButton = findViewById<Button>(R.id.predictDrawButton)
        val clearButton = findViewById<TextView>(R.id.clearDrawButton)


        resultText = findViewById(R.id.drawResultText)
        previewImageView = findViewById(R.id.drawPreviewImageView)

        drawingView = DigitDrawingView(this)
        drawingContainer.addView(
            drawingView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )

        interpreter = Interpreter(loadModelFile())

        predictButton.setOnClickListener {
            val drawnBitmap = drawingView.getBitmapCopy()
            val processed28 = preprocessDrawnDigit(drawnBitmap)

            if (processed28 == null) {
                resultText.text = "Please draw one digit first"
                previewImageView.setImageDrawable(null)
                return@setOnClickListener
            }

            previewImageView.setImageBitmap(
                Bitmap.createScaledBitmap(processed28, 140, 140, false)
            )

            val input = bitmapToFloatBuffer(processed28)
            val output = Array(1) { FloatArray(10) }

            input.rewind()
            interpreter.run(input, output)

            val predictedDigit = output[0].indices.maxByOrNull { output[0][it] } ?: -1
            val confidence = if (predictedDigit != -1) output[0][predictedDigit] else 0f

            resultText.text = String.format(
                Locale.US,
                "Prediction: %d\nConfidence: %.2f",
                predictedDigit,
                confidence
            )
        }

        clearButton.setOnClickListener {
            drawingView.clear()
            previewImageView.setImageDrawable(null)
            resultText.text = "Draw a digit, then press Predict"
        }
    }

    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = assets.openFd("arabic_digit_mlp_model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel

        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun preprocessDrawnDigit(bitmap: Bitmap): Bitmap? {
        val width = bitmap.width
        val height = bitmap.height

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        var minX = width
        var minY = height
        var maxX = 0
        var maxY = 0
        var foundDigit = false

        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]
                val value = Color.red(pixel)

                if (value > WHITE_THRESHOLD) {
                    minX = minOf(minX, x)
                    minY = minOf(minY, y)
                    maxX = maxOf(maxX, x)
                    maxY = maxOf(maxY, y)
                    foundDigit = true
                }
            }
        }

        if (!foundDigit) return null

        val cropWidth = maxX - minX + 1
        val cropHeight = maxY - minY + 1

        val cropPixels = IntArray(cropWidth * cropHeight) { Color.BLACK }

        for (y in 0 until cropHeight) {
            for (x in 0 until cropWidth) {
                val sourceX = minX + x
                val sourceY = minY + y
                val pixel = pixels[sourceY * width + sourceX]
                val value = Color.red(pixel)

                if (value > WHITE_THRESHOLD) {
                    cropPixels[y * cropWidth + x] = Color.rgb(value, value, value)
                }
            }
        }

        val cropBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Bitmap.Config.ARGB_8888)
        cropBitmap.setPixels(cropPixels, 0, cropWidth, 0, 0, cropWidth, cropHeight)

        val maxDim = maxOf(cropWidth, cropHeight)
        val scale = DIGIT_SIZE.toFloat() / maxDim.toFloat()
        val resizedWidth = maxOf(1, (cropWidth * scale).roundToInt())
        val resizedHeight = maxOf(1, (cropHeight * scale).roundToInt())

        val resized = Bitmap.createScaledBitmap(cropBitmap, resizedWidth, resizedHeight, true)
        val resizedPixels = IntArray(resizedWidth * resizedHeight)
        resized.getPixels(resizedPixels, 0, resizedWidth, 0, 0, resizedWidth, resizedHeight)

        val finalPixels = IntArray(TARGET_SIZE * TARGET_SIZE) { Color.BLACK }
        val pasteLeft = (TARGET_SIZE - resizedWidth) / 2
        val pasteTop = (TARGET_SIZE - resizedHeight) / 2

        for (y in 0 until resizedHeight) {
            for (x in 0 until resizedWidth) {
                finalPixels[(pasteTop + y) * TARGET_SIZE + pasteLeft + x] =
                    resizedPixels[y * resizedWidth + x]
            }
        }

        val centeredPixels = centerByMass(finalPixels, TARGET_SIZE, TARGET_SIZE)
        val final28 = Bitmap.createBitmap(TARGET_SIZE, TARGET_SIZE, Bitmap.Config.ARGB_8888)
        final28.setPixels(centeredPixels, 0, TARGET_SIZE, 0, 0, TARGET_SIZE, TARGET_SIZE)

        return final28
    }

    private fun centerByMass(pixels: IntArray, width: Int, height: Int): IntArray {
        var sum = 0f
        var sumX = 0f
        var sumY = 0f

        for (y in 0 until height) {
            for (x in 0 until width) {
                val value = Color.red(pixels[y * width + x]).toFloat()

                if (value > 0f) {
                    sum += value
                    sumX += x * value
                    sumY += y * value
                }
            }
        }

        if (sum <= 0f) return pixels

        val centerX = sumX / sum
        val centerY = sumY / sum
        val shiftX = (width / 2f - centerX).roundToInt()
        val shiftY = (height / 2f - centerY).roundToInt()

        val out = IntArray(width * height) { Color.BLACK }

        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = pixels[y * width + x]

                if (Color.red(pixel) == 0) continue

                val nx = x + shiftX
                val ny = y + shiftY

                if (nx in 0 until width && ny in 0 until height) {
                    out[ny * width + nx] = pixel
                }
            }
        }

        return out
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(1 * TARGET_SIZE * TARGET_SIZE * 1 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(TARGET_SIZE * TARGET_SIZE)
        bitmap.getPixels(pixels, 0, TARGET_SIZE, 0, 0, TARGET_SIZE, TARGET_SIZE)

        for (pixel in pixels) {
            inputBuffer.putFloat(Color.red(pixel) / 255.0f)
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter.close()
    }
}

class DigitDrawingView(context: Context) : View(context) {

    private val path = Path()
    private var lastX = 0f
    private var lastY = 0f

    private val drawPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 42f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private lateinit var bitmap: Bitmap
    private lateinit var bitmapCanvas: Canvas

    init {
        setBackgroundColor(Color.BLACK)
    }

    override fun onSizeChanged(width: Int, height: Int, oldWidth: Int, oldHeight: Int) {
        super.onSizeChanged(width, height, oldWidth, oldHeight)

        if (width > 0 && height > 0) {
            bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bitmapCanvas = Canvas(bitmap)
            bitmapCanvas.drawColor(Color.BLACK)
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawBitmap(bitmap, 0f, 0f, null)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        parent?.requestDisallowInterceptTouchEvent(true)

        val x = event.x
        val y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                path.reset()
                path.moveTo(x, y)

                lastX = x
                lastY = y

                invalidate()
                return true
            }

            MotionEvent.ACTION_MOVE -> {
                val midX = (lastX + x) / 2f
                val midY = (lastY + y) / 2f

                path.quadTo(lastX, lastY, midX, midY)
                bitmapCanvas.drawPath(path, drawPaint)

                lastX = x
                lastY = y

                invalidate()
                return true
            }

            MotionEvent.ACTION_UP -> {
                path.lineTo(x, y)
                bitmapCanvas.drawPath(path, drawPaint)
                path.reset()

                parent?.requestDisallowInterceptTouchEvent(false)

                invalidate()
                return true
            }

            MotionEvent.ACTION_CANCEL -> {
                path.reset()
                parent?.requestDisallowInterceptTouchEvent(false)
                invalidate()
                return true
            }
        }

        return true
    }

    fun clear() {
        bitmapCanvas.drawColor(Color.BLACK)
        invalidate()
    }

    fun getBitmapCopy(): Bitmap {
        return bitmap.copy(Bitmap.Config.ARGB_8888, false)
    }
}
