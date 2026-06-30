package com.example.arabicdigitsprediction

import android.content.Intent
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class WelcomeActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_welcome)

        val getStartedButton = findViewById<TextView>(R.id.getStartedButton)

        getStartedButton.setOnClickListener {
            startActivity(Intent(this, ModeSelectionActivity::class.java))
        }
    }
}
