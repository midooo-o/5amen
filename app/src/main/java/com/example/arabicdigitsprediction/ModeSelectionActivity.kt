package com.example.arabicdigitsprediction

import android.content.Intent
import android.os.Bundle
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity

class ModeSelectionActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_mode_selection)

        val uploadOption = findViewById<LinearLayout>(R.id.uploadOption)
        val drawOption = findViewById<LinearLayout>(R.id.drawOption)

        uploadOption.setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
        }

        drawOption.setOnClickListener {
            startActivity(Intent(this, DrawActivity::class.java))
        }
    }
}
