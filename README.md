# 😊 Facial Expression Detector

I am excited to share that this is a facial expression recognition web app built with **Streamlit**, **OpenCV**, and a **pre-trained Hugging Face image classification model**. It detects faces in uploaded images and classifies emotional states like happy, sad, angry, neutral, etc.

---

## 🚀 Live Demo

Try it now: 🌐 [facial-expression-detector.streamlit.app](https://facial-expression-detector.streamlit.app/)

---

## 🧠 How It Works

- 🧍 **Face Detection**: Uses OpenCV’s Haar Cascade classifier to detect faces in an image.
- 🎯 **Emotion Prediction**: Cropped face regions are passed to a Hugging Face image classifier — [`dima806/facial_emotions_image_detection`](https://huggingface.co/dima806/facial_emotions_image_detection).
- 🖼️ **Annotation**: Streamlit overlays predicted emotions and confidence scores on each detected face.

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io/) – Interactive UI
- [OpenCV](https://opencv.org/) – Face detection
- [Hugging Face Transformers](https://huggingface.co/transformers/) – Pretrained model
- [PyTorch](https://pytorch.org/) – Backend for inference
- [Pillow](https://pillow.readthedocs.io/) – Image handling

---

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/M-Hamza-Hassaan/Facial-Expression-Detector.git
cd Facial-Expression-Detector
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the app:
```
streamlit run app.py
```

### How to Enhance This App, Here are ideas for how you can take this app further:


🧠 Add more fine-grained or custom-trained emotion categories

📊 Log predictions for analytics (e.g., in classrooms or meetings)

🎨 Improve UI (e.g., dark mode, animated transitions)

---

[You can Follow me on LinkedIn](https://www.linkedin.com/in/muhammad-hamza-hassaan-29920a25a/)
