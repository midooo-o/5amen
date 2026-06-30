package com.example.arabicdigitsprediction

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.ArrayDeque
import java.util.Locale
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TARGET_SIZE = 28
        private const val DIGIT_SIZE = 20
        private const val MAX_PREPROCESS_SIDE = 1000
        private const val LOW_CONFIDENCE_THRESHOLD = 0.70f

        // Blue pen mask tuning
        private const val BLUE_HUE_MIN = 180f
        private const val BLUE_HUE_MAX = 285f
        private const val MIN_BLUE_SATURATION = 0.18f
        private const val STRONG_BLUE_SATURATION = 0.32f
        private const val MIN_BLUE_DOMINANCE = 10
        private const val MIN_CHROMA = 18
        private const val PAPER_DARKNESS_GAP = 0.05f

        // Segmentation tuning
        private const val MORPH_RADIUS = 1
        private const val MIN_COMPONENT_AREA_PIXELS = 8
        private const val MIN_COMPONENT_AREA_FRACTION = 0.00001f
        private const val MAX_COMPONENT_AREA_FRACTION = 0.35f

        private const val MIN_DIGIT_WIDTH_PIXELS = 3
        private const val MIN_DIGIT_HEIGHT_PIXELS = 3
        private const val MIN_DIGIT_SIZE_FRACTION = 0.003f

        private const val MIN_ASPECT_RATIO = 0.08f
        private const val MAX_ASPECT_RATIO = 6.0f

        // Merge only very nearby split pieces of the same digit.
        private const val MERGE_GAP_PIXELS = 4
        private const val MERGE_GAP_FRACTION = 0.06f
        private const val MERGE_OVERLAP_RATIO = 0.12f

        private const val DEBUG_DIGIT_SCALE = 4
    }

    private lateinit var imageView: ImageView
    private lateinit var debugImageView: ImageView
    private lateinit var resultText: TextView
    private lateinit var interpreter: Interpreter

    private var selectedBitmap: Bitmap? = null

    private data class DigitBox(
        val left: Int,
        val top: Int,
        val right: Int,
        val bottom: Int,
        val area: Int
    ) {
        val width: Int get() = right - left + 1
        val height: Int get() = bottom - top + 1

        fun merge(other: DigitBox): DigitBox {
            return DigitBox(
                left = minOf(left, other.left),
                top = minOf(top, other.top),
                right = maxOf(right, other.right),
                bottom = maxOf(bottom, other.bottom),
                area = area + other.area
            )
        }
    }

    private data class SegmentationResult(
        val sourceBitmap: Bitmap,
        val mask: BooleanArray,
        val width: Int,
        val height: Int,
        val boxes: List<DigitBox>
    )

    private data class DigitPrediction(
        val digit: Int,
        val confidence: Float,
        val box: DigitBox,
        val processedBitmap: Bitmap
    )

    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) {
                selectedBitmap = contentResolver.openInputStream(uri)?.use {
                    BitmapFactory.decodeStream(it)
                }

                if (selectedBitmap != null) {
                    imageView.setImageBitmap(selectedBitmap)
                    debugImageView.visibility = View.GONE
                    resultText.text = "Image selected. Press Predict."
                } else {
                    resultText.text = "Could not decode image"
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        debugImageView = findViewById(R.id.debugImageView)
        resultText = findViewById(R.id.resultText)

        val pickImageButton = findViewById<Button>(R.id.pickImageButton)
        val predictButton = findViewById<Button>(R.id.predictButton)
        val clearButton = findViewById<TextView>(R.id.clearButton)
        interpreter = Interpreter(loadModelFile())

        pickImageButton.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        predictButton.setOnClickListener {
            val bitmap = selectedBitmap

            if (bitmap == null) {
                resultText.text = "Please pick an image first"
                return@setOnClickListener
            }

            val segmentation = segmentDigits(bitmap)

            if (segmentation.boxes.isEmpty()) {
                imageView.setImageBitmap(segmentation.sourceBitmap)
                debugImageView.visibility = View.GONE
                resultText.text = "No digits detected"
                return@setOnClickListener
            }

            val predictions = ArrayList<DigitPrediction>()

            for (box in segmentation.boxes) {
                val processed28 = renderMaskTo28(
                    mask = segmentation.mask,
                    width = segmentation.width,
                    height = segmentation.height,
                    box = box
                )

                val input = bitmapToFloatBuffer(processed28)
                val output = Array(1) { FloatArray(10) }

                input.rewind()
                interpreter.run(input, output)

                val predictedDigit = output[0].indices.maxByOrNull { output[0][it] } ?: -1
                val confidence = if (predictedDigit != -1) output[0][predictedDigit] else 0f

                predictions.add(
                    DigitPrediction(
                        digit = predictedDigit,
                        confidence = confidence,
                        box = box,
                        processedBitmap = processed28
                    )
                )
            }

            imageView.setImageBitmap(drawBoundingBoxes(segmentation.sourceBitmap, predictions))
            debugImageView.setImageBitmap(createProcessedDigitsPreview(predictions))
            debugImageView.visibility = View.VISIBLE
            resultText.text = buildResultText(predictions)
        }

        clearButton.setOnClickListener {
            selectedBitmap = null
            imageView.setImageDrawable(null)
            debugImageView.setImageDrawable(null)
            debugImageView.visibility = View.GONE
            resultText.text = "Result will appear here"
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

    private fun segmentDigits(bitmap: Bitmap): SegmentationResult {
        val source = bitmapForPreprocess(bitmap)
        val width = source.width
        val height = source.height

        val pixels = IntArray(width * height)
        source.getPixels(pixels, 0, width, 0, 0, width, height)

        val paperValue = estimatePaperValue(pixels)
        val rawMask = BooleanArray(width * height)
        val hsv = FloatArray(3)

        for (i in pixels.indices) {
            rawMask[i] = isBluePenPixel(pixels[i], paperValue, hsv)
        }

        val cleanedMask = closeMask(rawMask, width, height, MORPH_RADIUS)
        val components = findConnectedComponents(cleanedMask, width, height)

        val candidateBoxes = components.filter {
            isCandidateComponent(it, width, height)
        }

        val mergedBoxes = mergeNearbyBoxes(candidateBoxes)

        val finalBoxes = mergedBoxes
            .filter { isValidDigitBox(it, width, height) }
            .sortedBy { it.left }

        val finalMask = keepMaskInsideBoxes(cleanedMask, width, height, finalBoxes)

        return SegmentationResult(
            sourceBitmap = source,
            mask = finalMask,
            width = width,
            height = height,
            boxes = finalBoxes
        )
    }

    private fun isBluePenPixel(pixel: Int, paperValue: Float, hsv: FloatArray): Boolean {
        val r = Color.red(pixel)
        val g = Color.green(pixel)
        val b = Color.blue(pixel)

        Color.colorToHSV(pixel, hsv)

        val hue = hsv[0]
        val saturation = hsv[1]
        val value = hsv[2]

        val maxChannel = maxOf(r, g, b)
        val minChannel = minOf(r, g, b)
        val chroma = maxChannel - minChannel
        val blueDominance = b - maxOf(r, g)

        val hueIsBlue = hue in BLUE_HUE_MIN..BLUE_HUE_MAX
        val hasEnoughColor = saturation >= MIN_BLUE_SATURATION && chroma >= MIN_CHROMA
        val isBlueDominant =
            blueDominance >= MIN_BLUE_DOMINANCE ||
                    (saturation >= STRONG_BLUE_SATURATION && b > r && b > g)

        val isNotPaper =
            value <= paperValue - PAPER_DARKNESS_GAP ||
                    saturation >= STRONG_BLUE_SATURATION

        return hueIsBlue && hasEnoughColor && isBlueDominant && isNotPaper
    }

    private fun bitmapForPreprocess(bitmap: Bitmap): Bitmap {
        val maxSide = maxOf(bitmap.width, bitmap.height)
        val scale = if (maxSide > MAX_PREPROCESS_SIDE) {
            MAX_PREPROCESS_SIDE.toFloat() / maxSide.toFloat()
        } else {
            1f
        }

        val scaled = if (scale < 1f) {
            Bitmap.createScaledBitmap(
                bitmap,
                maxOf(1, (bitmap.width * scale).roundToInt()),
                maxOf(1, (bitmap.height * scale).roundToInt()),
                true
            )
        } else {
            bitmap
        }

        return if (scaled.config == Bitmap.Config.ARGB_8888) {
            scaled
        } else {
            scaled.copy(Bitmap.Config.ARGB_8888, false)
        }
    }

    private fun estimatePaperValue(pixels: IntArray): Float {
        val hist = IntArray(256)
        val step = maxOf(1, pixels.size / 50000)

        var samples = 0
        var i = 0
        while (i < pixels.size) {
            val pixel = pixels[i]
            val value = maxOf(Color.red(pixel), Color.green(pixel), Color.blue(pixel))
            hist[value]++
            samples++
            i += step
        }

        val target = (samples * 0.85f).roundToInt().coerceAtLeast(1)
        var count = 0

        for (value in 0..255) {
            count += hist[value]
            if (count >= target) return value / 255f
        }

        return 1f
    }

    private fun closeMask(mask: BooleanArray, width: Int, height: Int, radius: Int): BooleanArray {
        if (radius <= 0) return mask.copyOf()
        return erodeMask(dilateMask(mask, width, height, radius), width, height, radius)
    }

    private fun dilateMask(mask: BooleanArray, width: Int, height: Int, radius: Int): BooleanArray {
        val out = BooleanArray(mask.size)

        for (y in 0 until height) {
            for (x in 0 until width) {
                var on = false

                search@ for (yy in maxOf(0, y - radius)..minOf(height - 1, y + radius)) {
                    for (xx in maxOf(0, x - radius)..minOf(width - 1, x + radius)) {
                        if (mask[yy * width + xx]) {
                            on = true
                            break@search
                        }
                    }
                }

                out[y * width + x] = on
            }
        }

        return out
    }

    private fun erodeMask(mask: BooleanArray, width: Int, height: Int, radius: Int): BooleanArray {
        val out = BooleanArray(mask.size)

        for (y in 0 until height) {
            for (x in 0 until width) {
                var on = true

                search@ for (yy in maxOf(0, y - radius)..minOf(height - 1, y + radius)) {
                    for (xx in maxOf(0, x - radius)..minOf(width - 1, x + radius)) {
                        if (!mask[yy * width + xx]) {
                            on = false
                            break@search
                        }
                    }
                }

                out[y * width + x] = on
            }
        }

        return out
    }

    private fun findConnectedComponents(
        mask: BooleanArray,
        width: Int,
        height: Int
    ): List<DigitBox> {
        val visited = BooleanArray(mask.size)
        val components = ArrayList<DigitBox>()
        val queue = ArrayDeque<Int>()

        for (start in mask.indices) {
            if (!mask[start] || visited[start]) continue

            var area = 0
            var minX = width
            var minY = height
            var maxX = 0
            var maxY = 0

            visited[start] = true
            queue.clear()
            queue.add(start)

            while (!queue.isEmpty()) {
                val index = queue.removeFirst()
                val x = index % width
                val y = index / width

                area++
                minX = minOf(minX, x)
                minY = minOf(minY, y)
                maxX = maxOf(maxX, x)
                maxY = maxOf(maxY, y)

                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue

                        val nx = x + dx
                        val ny = y + dy

                        if (nx !in 0 until width || ny !in 0 until height) continue

                        val next = ny * width + nx
                        if (mask[next] && !visited[next]) {
                            visited[next] = true
                            queue.add(next)
                        }
                    }
                }
            }

            components.add(DigitBox(minX, minY, maxX, maxY, area))
        }

        return components
    }

    private fun isCandidateComponent(box: DigitBox, width: Int, height: Int): Boolean {
        val totalPixels = width * height
        val minArea = maxOf(
            MIN_COMPONENT_AREA_PIXELS,
            (totalPixels * MIN_COMPONENT_AREA_FRACTION).roundToInt()
        )
        val maxArea = maxOf(
            minArea + 1,
            (totalPixels * MAX_COMPONENT_AREA_FRACTION).roundToInt()
        )

        val aspectRatio = box.width.toFloat() / box.height.toFloat()
        val touchesBorder =
            box.left <= 1 || box.top <= 1 || box.right >= width - 2 || box.bottom >= height - 2
        val looksLikeBorderLine =
            touchesBorder && (box.width > width * 0.50f || box.height > height * 0.50f)

        return box.area in minArea..maxArea &&
                aspectRatio in 0.03f..20.0f &&
                !looksLikeBorderLine
    }

    private fun isValidDigitBox(box: DigitBox, width: Int, height: Int): Boolean {
        val totalPixels = width * height
        val minArea = maxOf(
            MIN_COMPONENT_AREA_PIXELS,
            (totalPixels * MIN_COMPONENT_AREA_FRACTION).roundToInt()
        )
        val maxArea = maxOf(
            minArea + 1,
            (totalPixels * MAX_COMPONENT_AREA_FRACTION).roundToInt()
        )

        val minWidth = maxOf(
            MIN_DIGIT_WIDTH_PIXELS,
            (width * MIN_DIGIT_SIZE_FRACTION).roundToInt()
        )
        val minHeight = maxOf(
            MIN_DIGIT_HEIGHT_PIXELS,
            (height * MIN_DIGIT_SIZE_FRACTION).roundToInt()
        )

        val aspectRatio = box.width.toFloat() / box.height.toFloat()
        val touchesBorder =
            box.left <= 1 || box.top <= 1 || box.right >= width - 2 || box.bottom >= height - 2
        val hugeBorderObject =
            touchesBorder && (box.width > width * 0.60f || box.height > height * 0.60f)

        return box.area in minArea..maxArea &&
                box.width >= minWidth &&
                box.height >= minHeight &&
                aspectRatio in MIN_ASPECT_RATIO..MAX_ASPECT_RATIO &&
                !hugeBorderObject
    }

    private fun mergeNearbyBoxes(inputBoxes: List<DigitBox>): List<DigitBox> {
        if (inputBoxes.isEmpty()) return emptyList()

        var boxes = inputBoxes.sortedBy { it.left }
        var changed: Boolean

        do {
            changed = false
            val used = BooleanArray(boxes.size)
            val next = ArrayList<DigitBox>()

            for (i in boxes.indices) {
                if (used[i]) continue

                var current = boxes[i]
                used[i] = true

                var mergedOne: Boolean
                do {
                    mergedOne = false

                    for (j in boxes.indices) {
                        if (used[j]) continue

                        if (shouldMergeBoxes(current, boxes[j])) {
                            current = current.merge(boxes[j])
                            used[j] = true
                            changed = true
                            mergedOne = true
                        }
                    }
                } while (mergedOne)

                next.add(current)
            }

            boxes = next.sortedBy { it.left }
        } while (changed)

        return boxes
    }

    private fun shouldMergeBoxes(a: DigitBox, b: DigitBox): Boolean {
        val gapX = horizontalGap(a, b)
        val gapY = verticalGap(a, b)

        val overlapX = overlapX(a, b)
        val overlapY = overlapY(a, b)

        val minWidth = minOf(a.width, b.width).coerceAtLeast(1)
        val minHeight = minOf(a.height, b.height).coerceAtLeast(1)

        val overlapXRatio = overlapX.toFloat() / minWidth.toFloat()
        val overlapYRatio = overlapY.toFloat() / minHeight.toFloat()

        val allowedGapX = maxOf(
            MERGE_GAP_PIXELS,
            (minHeight * MERGE_GAP_FRACTION).roundToInt()
        )
        val allowedGapY = maxOf(
            MERGE_GAP_PIXELS,
            (minWidth * MERGE_GAP_FRACTION).roundToInt()
        )

        val closeSideBySide = gapX <= allowedGapX && overlapYRatio >= MERGE_OVERLAP_RATIO
        val closeStacked = gapY <= allowedGapY && overlapXRatio >= MERGE_OVERLAP_RATIO
        val almostTouchingCorners = gapX <= MERGE_GAP_PIXELS && gapY <= MERGE_GAP_PIXELS

        val combined = a.merge(b)
        val tooWideForOneDigit =
            combined.width > maxOf(a.height, b.height) * 1.8f && gapX > MERGE_GAP_PIXELS

        return !tooWideForOneDigit && (closeSideBySide || closeStacked || almostTouchingCorners)
    }

    private fun horizontalGap(a: DigitBox, b: DigitBox): Int {
        return when {
            a.right < b.left -> b.left - a.right - 1
            b.right < a.left -> a.left - b.right - 1
            else -> 0
        }
    }

    private fun verticalGap(a: DigitBox, b: DigitBox): Int {
        return when {
            a.bottom < b.top -> b.top - a.bottom - 1
            b.bottom < a.top -> a.top - b.bottom - 1
            else -> 0
        }
    }

    private fun overlapX(a: DigitBox, b: DigitBox): Int {
        return maxOf(0, minOf(a.right, b.right) - maxOf(a.left, b.left) + 1)
    }

    private fun overlapY(a: DigitBox, b: DigitBox): Int {
        return maxOf(0, minOf(a.bottom, b.bottom) - maxOf(a.top, b.top) + 1)
    }

    private fun keepMaskInsideBoxes(
        mask: BooleanArray,
        width: Int,
        height: Int,
        boxes: List<DigitBox>
    ): BooleanArray {
        val out = BooleanArray(mask.size)

        for (box in boxes) {
            val left = box.left.coerceIn(0, width - 1)
            val top = box.top.coerceIn(0, height - 1)
            val right = box.right.coerceIn(0, width - 1)
            val bottom = box.bottom.coerceIn(0, height - 1)

            for (y in top..bottom) {
                for (x in left..right) {
                    val index = y * width + x
                    if (mask[index]) out[index] = true
                }
            }
        }

        return out
    }

    private fun renderMaskTo28(
        mask: BooleanArray,
        width: Int,
        height: Int,
        box: DigitBox
    ): Bitmap {
        val left = box.left.coerceIn(0, width - 1)
        val top = box.top.coerceIn(0, height - 1)
        val right = box.right.coerceIn(0, width - 1)
        val bottom = box.bottom.coerceIn(0, height - 1)

        val cropWidth = right - left + 1
        val cropHeight = bottom - top + 1

        val cropPixels = IntArray(cropWidth * cropHeight) { Color.BLACK }

        for (y in 0 until cropHeight) {
            for (x in 0 until cropWidth) {
                val sourceX = left + x
                val sourceY = top + y

                if (mask[sourceY * width + sourceX]) {
                    cropPixels[y * cropWidth + x] = Color.WHITE
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
            val value = Color.red(pixel) / 255.0f
            inputBuffer.putFloat(value)
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    private fun drawBoundingBoxes(
        sourceBitmap: Bitmap,
        predictions: List<DigitPrediction>
    ): Bitmap {
        val preview = sourceBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(preview)

        val strokeWidth = maxOf(3f, maxOf(preview.width, preview.height) / 250f)

        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(220, 40, 40)
            style = Paint.Style.STROKE
            this.strokeWidth = strokeWidth
        }

        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = maxOf(24f, preview.width / 35f)
            typeface = Typeface.DEFAULT_BOLD
        }

        val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.rgb(220, 40, 40)
            style = Paint.Style.FILL
        }

        val textBounds = Rect()

        for (prediction in predictions) {
            val box = prediction.box

            canvas.drawRect(
                box.left.toFloat(),
                box.top.toFloat(),
                (box.right + 1).toFloat(),
                (box.bottom + 1).toFloat(),
                boxPaint
            )

            val label = if (prediction.confidence < LOW_CONFIDENCE_THRESHOLD) {
                "${prediction.digit}?"
            } else {
                prediction.digit.toString()
            }

            textPaint.getTextBounds(label, 0, label.length, textBounds)

            val labelLeft = box.left.toFloat()
            val labelTop = (box.top - textBounds.height() - 12).coerceAtLeast(0).toFloat()
            val labelRight = labelLeft + textBounds.width() + 18
            val labelBottom = labelTop + textBounds.height() + 14

            canvas.drawRect(labelLeft, labelTop, labelRight, labelBottom, labelBackgroundPaint)
            canvas.drawText(label, labelLeft + 9, labelBottom - 7, textPaint)
        }

        return preview
    }

    private fun createProcessedDigitsPreview(predictions: List<DigitPrediction>): Bitmap {
        val digitPreviewSize = TARGET_SIZE * DEBUG_DIGIT_SCALE
        val gap = 12
        val labelHeight = 34

        val width = maxOf(1, predictions.size * digitPreviewSize + (predictions.size + 1) * gap)
        val height = digitPreviewSize + labelHeight + gap * 2

        val preview = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(preview)
        canvas.drawColor(Color.WHITE)

        val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.LTGRAY
            style = Paint.Style.STROKE
            strokeWidth = 2f
        }

        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.BLACK
            textSize = 22f
            textAlign = Paint.Align.CENTER
            typeface = Typeface.DEFAULT_BOLD
        }

        for ((index, prediction) in predictions.withIndex()) {
            val left = gap + index * (digitPreviewSize + gap)
            val top = gap

            val scaledDigit = Bitmap.createScaledBitmap(
                prediction.processedBitmap,
                digitPreviewSize,
                digitPreviewSize,
                false
            )

            canvas.drawBitmap(scaledDigit, left.toFloat(), top.toFloat(), null)
            canvas.drawRect(
                left.toFloat(),
                top.toFloat(),
                (left + digitPreviewSize).toFloat(),
                (top + digitPreviewSize).toFloat(),
                borderPaint
            )

            val confidenceText = String.format(Locale.US, "%.2f", prediction.confidence)
            val label = "${prediction.digit}  $confidenceText"

            canvas.drawText(
                label,
                left + digitPreviewSize / 2f,
                (top + digitPreviewSize + 25).toFloat(),
                textPaint
            )
        }

        return preview
    }

    private fun buildResultText(predictions: List<DigitPrediction>): String {
        val finalPrediction = predictions.joinToString(separator = "") { it.digit.toString() }
        val builder = StringBuilder()

        builder.append("Final Prediction: ").append(finalPrediction).append("\n")
        builder.append("Detected digits: ").append(predictions.size).append("\n\n")
        builder.append("Confidence for each digit:\n")

        for ((index, prediction) in predictions.withIndex()) {
            val confidenceText = String.format(Locale.US, "%.2f", prediction.confidence)
            val lowConfidenceText = if (prediction.confidence < LOW_CONFIDENCE_THRESHOLD) {
                "  Low confidence"
            } else {
                ""
            }

            builder.append(index + 1)
                .append(". ")
                .append(prediction.digit)
                .append(" = ")
                .append(confidenceText)
                .append(lowConfidenceText)
                .append("\n")
        }

        return builder.toString().trim()
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter.close()
    }
}
