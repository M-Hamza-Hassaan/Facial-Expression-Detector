import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

MODEL_LOCAL_PATH = "./local_facial_emotion_model"
@st.cache_resource
def load_face_cascade():
    """Load the OpenCV Haar cascade for face detection."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_cascade


@st.cache_resource
def load_emotion_model():
    repo_id = "dima806/facial_emotions_image_detection"  # Load processor and model from the extracted directory
    processor = AutoImageProcessor.from_pretrained(repo_id)
    model = AutoModelForImageClassification.from_pretrained(repo_id)
    return processor, model

def detect_and_predict_emotions(image, face_cascade, processor, model):
    """
    Args:
        image: The input image (PIL or NumPy format)
        face_cascade: Haar cascade model for face detection
        processor: Image processor for the emotion recognition model
        model: Pre-trained emotion recognition model
    Detects faces in the image, crops them, and predicts emotions using the model.
    If no faces are detected, returns an empty result.
    If an error occurs during prediction, it returns an error message and draws a red box around the face.
    If successful, it returns the emotion prediction data and the annotated image.

    Returns:
        A tuple of results (emotion prediction data) and the annotated image in RGB
    """
    # Convert PIL image to NumPy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Ensure the image has 3 color channels for processing
    if image_np.shape[-1] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    annotated_image = image_bgr.copy()
    results = []

    for (x, y, w, h) in faces:
        # Define a padded region around the detected face
        pad = 20
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image_bgr.shape[1], x + w + pad), min(image_bgr.shape[0], y + h + pad)

        # Crop and convert the face region for prediction
        face_roi = image_bgr[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)


        try:
            st.image(face_pil, caption="Detected face")

            # Preprocess and predict
            inputs = processor(images=face_pil, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            # Map probabilities to emotion labels
            emotion_scores = {
                model.config.id2label[i]: float(score)
                for i, score in enumerate(probs[0])
            }

            # Get top prediction
            top_pred_index = torch.argmax(probs, dim=1).item()
            predicted_emotion = model.config.id2label[top_pred_index]
            confidence = probs[0][top_pred_index].item()

            # Show debug info
            st.write("All predicted scores:", emotion_scores)

            # Save result
            results.append({
                'emotion': predicted_emotion,
                'confidence': confidence,
                'probabilities': emotion_scores,
                'bbox': (x, y, w, h)
            })

            # Draw box and label
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            label = f"{predicted_emotion}: {confidence:.2%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                annotated_image,
                (x, y - 35),
                (x + label_size[0] + 10, y),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                annotated_image,
                label,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        except Exception as e:
            st.error(f"⚠️ Emotion prediction error: {e}")
            # Draw red box and "Error" label in case of failure
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated_image, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Convert BGR image back to RGB for display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return results, annotated_image_rgb



def display_results(results, annotated_image, original_label="Original Image"):
    """Display the results in the Streamlit UI."""
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(original_label)
        st.image(annotated_image, use_container_width=True)
    with col2:
        st.subheader("Detected Emotions")
        st.image(annotated_image, use_container_width=True)

    if results:
        st.markdown("---")
        st.subheader("Analysis Results")
        for i, res in enumerate(results):
            with st.expander(f"Face {i + 1} - {res['emotion']} ({res['confidence']:.2%})"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Predicted Emotion", res['emotion'])
                    st.metric("Confidence", f"{res['confidence']:.2%}")
                with col2:
                    st.subheader("Emotion Probabilities")
                    for emo, prob in res['probabilities'].items():
                        st.progress(prob, text=f"{emo}: {prob:.2%}")
    else:
        st.warning("No faces detected. Please try another image.")

def main():
    st.title("😊 Facial Expression Recognition App")
    st.markdown("---")
    st.sidebar.title("Settings")
    st.sidebar.markdown("Upload an image or use your webcam to detect facial expressions!")

    # Load models
    with st.spinner("🔄 Loading models..."):
        face_cascade = load_face_cascade()
        processor, model = load_emotion_model()

    # Stop if model failed to load
    if processor is None or model is None:

        st.error("❌ Failed to load the emotion recognition model. Please check the logs.")
        return
    st.success("✅ Models loaded successfully!")
    st.markdown("---")
    st.sidebar.subheader("Input Method")
    st.sidebar.markdown("Choose how to input your image:")
    st.sidebar.markdown("1. **Upload Image**: Upload a local image file.")
    st.sidebar.markdown("2. **Take Photo with Camera**: Use your webcam to capture a photo.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.markdown("""
    - **Model**: `facial_emotions_image_detection`
    - **Source**: Hugging Face
    - **Face Detection**: OpenCV Haar Cascade
    - **Emotion Recognition**: Vision Transformer or CNN
    - **Deployment**: Docker-ready, Hugging Face Spaces compatible
    """)
    st.sidebar.markdown("---")
    st.sidebar.subheader("About This App")
    st.sidebar.markdown("""
    This app uses computer vision and deep learning to detect faces and recognize emotions from images.
    - Detects faces using OpenCV Haar Cascade
    - Recognizes emotions using a Hugging Face model
    - Visualizes predictions with confidence and probabilities
    """)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Technical Details")
    st.sidebar.markdown("""
    - **Model**: `facial_emotions_image_detection` (via Hugging Face)
    - **Library**: Streamlit, OpenCV, Transformers
    - **Face Detection**: Haar Cascade
    - **Emotion Recognition**: Vision Transformer or CNN
    - **Deployment**: Docker-ready, Hugging Face Spaces compatible
    """)

    # Input method: Upload or Camera
    input_method = st.radio("Choose input method:", ["Upload Image", "Take Photo with Camera"], horizontal=True)

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

            with st.spinner("Analyzing..."):
                results, annotated_image = detect_and_predict_emotions(image, face_cascade, processor, model)

                display_results(results, annotated_image)

    elif input_method == "Take Photo with Camera":
        camera_photo = st.camera_input("Take a photo")
        if camera_photo:
            image = Image.open(camera_photo)
            st.subheader("Captured Image")
            st.image(image, use_container_width=True)

            with st.spinner("Analyzing..."):
                results, annotated_image = detect_and_predict_emotions(image, face_cascade, processor, model)
                display_results(results, annotated_image)

    # Info section
    st.markdown("---")
    st.subheader("About This App")
    st.markdown("""
    This facial expression recognition app uses computer vision and deep learning to:
    - Detect faces using OpenCV
    - Recognize emotions with a Hugging Face model
    - Visualize predictions with confidence and probabilities
    """)

    with st.expander("🔍 Technical Details"):
        st.markdown(f"""
        - **Model**: facial_emotions_image_detection (via Hugging Face)
        - **Library**: Streamlit, OpenCV, Transformers
        - **Face Detection**: Haar Cascade
        - **Emotion Recognition**: Vision Transformer or CNN
        - **Deployment**: Docker-ready, Hugging Face Spaces compatible
        """)

# ------------------------ Run App ------------------------
if __name__ == "__main__":
    main()
