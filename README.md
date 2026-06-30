# 5amen

An Android application for Arabic handwritten digit recognition powered by a Neural Network and TensorFlow Lite.

---

## Overview

5amen is a machine learning-powered Android application that recognizes handwritten Arabic digits in real time. The project combines model development, mobile deployment, and user interaction into a complete end-to-end AI application.

The Neural Network model was trained using TensorFlow and Keras, converted to TensorFlow Lite, and integrated into a Kotlin-based Android application to perform efficient on-device inference.

Users can either draw handwritten digits directly on the screen or upload digit images from their device gallery to receive instant predictions.

---

## Features

- Arabic handwritten digit recognition
- Draw digits directly within the application
- Upload digit images from the gallery
- Real-time on-device prediction
- TensorFlow Lite inference engine
- Native Android application built with Kotlin

---

## Tech Stack

### Machine Learning
- Python
- TensorFlow
- Keras
- TensorFlow Lite

### Mobile Development
- Kotlin
- Android Studio

---

## Project Structure

```
5amen/
│
├── app/            # Android application source code
├── model/          # Trained model and TensorFlow Lite model
├── notebook/       # Model training notebooks
├── test/           # Testing resources
├── gradle/
└── ...
```

---

## Getting Started

### Prerequisites

- Android Studio
- Android SDK
- Kotlin
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/midooo-o/5amen.git
```

Open the project in Android Studio, sync Gradle dependencies, then build and run the application on an Android device or emulator.

---

## How It Works

1. A Neural Network is trained on Arabic handwritten digit data using TensorFlow and Keras.
2. The trained model is converted to TensorFlow Lite.
3. The TensorFlow Lite model is embedded into the Android application.
4. Users draw a digit or upload an image.
5. The application performs real-time inference and predicts the handwritten digit.

---

## Future Improvements

- Improve model accuracy with larger datasets
- Support additional handwritten Arabic characters
- Enhance application UI/UX
- Performance optimization for low-end devices

---

## Author

**Mohamed Tamer**

AI & Data Science Student

Interested in Machine Learning, Information Retrieval, Cybersecurity, and Intelligent Systems.

---

## License

This project is intended for educational and portfolio purposes.
