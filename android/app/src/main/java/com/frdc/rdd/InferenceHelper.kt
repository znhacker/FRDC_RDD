package com.frdc.rdd

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

class InferenceHelper(context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open("model.onnx").readBytes()
        session = env.createSession(modelBytes)
    }

    fun run(bitmap: Bitmap): FloatArray {
        val input = preprocess(bitmap)
        val tensor = OnnxTensor.createTensor(env, input, longArrayOf(1, 3, bitmap.height.toLong(), bitmap.width.toLong()))
        val results = session.run(mapOf(session.inputNames.iterator().next() to tensor))
        val output = results[0].value as Array<FloatArray>
        results.close()
        tensor.close()
        return output.flatten().toFloatArray()
    }

    private fun preprocess(bitmap: Bitmap): FloatBuffer {
        val w = bitmap.width
        val h = bitmap.height
        val floatBuffer = FloatBuffer.allocate(3 * w * h)
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val px = pixels[y * w + x]
                val r = (px shr 16 and 0xFF) / 255f
                val g = (px shr 8 and 0xFF) / 255f
                val b = (px and 0xFF) / 255f
                floatBuffer.put(r)
                floatBuffer.put(g)
                floatBuffer.put(b)
            }
        }
        floatBuffer.rewind()
        return floatBuffer
    }
}
