# WANI-MARK1
# 🖐️ Indian Sign Language (ISL) Recognition using Transfer Learning

This project implements a **deep learning model** to recognize **Indian Sign Language (ISL) letters and numbers** from images.
The model uses **Transfer Learning (MobileNetV2)** in **TensorFlow/Keras** for high accuracy and efficiency.

---

## 📂 Dataset

* The dataset consists of **folders of images**, one for each **letter/number** (e.g., `A/`, `B/`, `0/`, `1/`).
* Images are preprocessed to `128x128` pixels before training.
* Train/Test split: **80/20**.

---

## 🧠 Model Architecture

* **Base Model:** MobileNetV2 (pretrained on ImageNet, frozen initially).
* **Custom Classifier:**

  * Global Average Pooling
  * Dense (128 neurons, ReLU)
  * Dropout (0.5)
  * Dense (`num_classes`, Softmax)

---

## ⚙️ Training

* **Loss:** Sparse Categorical Crossentropy
* **Optimizer:** Adam
* **Metrics:** Accuracy
* **Epochs:** 10 (can be increased for fine-tuning)

---

## 🚀 Features

* Train ISL model from scratch or using Transfer Learning.
* Test predictions on **single images**.
* Capture **live webcam input** (Colab snapshot or local OpenCV stream).
* Save & load model in **native Keras format (`.keras`)**.

---

## 📊 Results

* The model achieves high accuracy on the ISL dataset (letters + numbers).
* Accuracy and loss graphs are generated after training.

---

## 🔍 Usage

### 1️⃣ Install dependencies

```bash
pip install tensorflow matplotlib split-folders opencv-python
```

### 2️⃣ Load Dataset

Put your dataset in:

```
ISL_dataset/
   A/
   B/
   ...
   0/
   1/
   ...
```

### 3️⃣ Train the Model

```python
python train.py
```

### 4️⃣ Test on an Image

```python
python predict.py --image path_to_image.jpg
```

### 5️⃣ Live Webcam (local only)

```python
python webcam_test.py
```

---

## 💾 Saving & Loading Model

Save:

```python
model.save("isl_sign_model.keras")
```

Load:

```python
from tensorflow import keras
model = keras.models.load_model("isl_sign_model.keras")
```

---

## 📸 Example Prediction

```text
Input Image: sign_A.jpg
Predicted: A
```

---

## 🔮 Future Improvements

* Add **fine-tuning** (unfreeze last layers of MobileNetV2).
* Support **real-time translation** (word-level ISL sentences).
* Deploy as a **Flask/Django Web App**.

---

## 🙌 Contributors

* **NIHAR KOCHE** – Developer & Researcher


