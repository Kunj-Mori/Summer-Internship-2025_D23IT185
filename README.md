# 🧠 Internship Project – AI-Powered Applications (May 2025)

This repository showcases the work I completed during my internship in May 2025. It includes four major AI/ML-based tasks: Text-to-Image Generation, Sentiment Analysis using NLP, Spam Detection using SVM, and Image/Video Enhancement with Cinematic Filter.

---

## 📅 Timeline

| Date        | Activity                                                                 |
|-------------|--------------------------------------------------------------------------|
| 12/05/2025  | Studied NumPy, Pandas, Keras, TensorFlow, PyTorch                        |
| 13/05/2025  | Continued exploration of ML/DL libraries                                 |
| 14/05/2025  | ✅ Completed Task 1: Text-to-Image Generator using Stable Diffusion      |
| 15/05/2025  | ✅ Completed Task 2: Sentiment Analysis using NLP                        |
| 16/05/2025  | ✅ Completed Task 3: Email Spam Classification using SVM                 |
| 17/05/2025  | 📌 Started Task 4: Researched Image/Video Enhancement techniques         |
| 18/05/2025  | ✅ Implemented cinematic filter for Image Enhancement                    |
| 19/05/2025  | ✅ Extended functionality to Video & Webcam with Streamlit UI            |
| 20/05/2025  | ✅ Enhanced video output and reduced pixel distortion                    |
| 22/05/2025  | ✅ Completed real-time enhancement using webcam                          |
| 23/05/2025  | ✅ Deployed full Cinematic Enhancement Project (image, video, webcam) via Streamlit Cloud |
| 24/05/2025  | 📌 Researched and addressed real-time frame capture issues in deployment |
| 25/05/2025  | ✅ Implemented Hugging Face API with predefined model for image enhancement |
| 26/05/2025  | 📌 Researched techniques to improve post-enhancement image quality       |
| 27/05/2025  | ✅ Improved quality and object accuracy in enhanced outputs              |
| 28/05/2025  | ✅ Used YOLO-based object detection and subject masking with `rembg` for refined cinematic effect |

---

## 🔧 Task 1: Text-to-Image Generation 🎨

**Description**:  
Created a web app using Streamlit to generate images from natural language prompts using Stable Diffusion v2.1.

**Model Used**: `stabilityai/stable-diffusion-2-1`  
**Libraries**: Streamlit, Torch, Diffusers, PIL, GC, Platform

**Features**:

- Interactive UI with adjustable parameters (image size, steps, guidance scale, seed)
- GPU/CPU optimized image generation
- Downloadable output
- Advanced settings with explanations
- Prompt & negative prompt support

---

## 💬 Task 2: Sentiment Analysis using NLP 🧾

**Description**:  
Built a sentiment classifier to label input text as positive or negative using Naive Bayes.

**Libraries Used**:  
Scikit-learn

**Process**:

- Collected 8 sample labeled texts
- Used CountVectorizer for feature extraction
- Trained MultinomialNB classifier
- Predicted sentiment of user-provided prompt

---

## 📧 Task 3: Spam Classification using SVM 🚫

**Description**:  
Developed a simple SVM-based email classifier to detect Spam vs Not Spam.

**Libraries Used**:  
Scikit-learn, re (Regex)

**Process**:

- Preprocessed emails using regex (lowercase, remove non-alphanumeric)
- Used CountVectorizer to vectorize the text
- Trained SVC with linear kernel
- Predicted category of user input email

---

## 🎞️ Task 4: Image & Video Enhancement – Cinematic Filter ✨

**Description**:  
Designed and implemented a Cinematic Enhancement Filter for images, videos, and webcam feed. The filter enhances media using contrast, brightness, color grading, tinting, vignette effect, and film grain. All components were unified into a web app using Streamlit, with real-time deployment support.

**Libraries Used**:  
OpenCV, Pillow, NumPy, Matplotlib, Google Colab, Streamlit, rembg, Hugging Face API

**Key Features**:

- Applies cinematic enhancements to **images, videos, and webcam feed**
- Adjustable parameters for fine-tuning
- Real-time processing with webcam
- Streamlit Cloud deployment support
- Hugging Face API integration for fast, pretrained image enhancement
- Image quality refinement via post-processing and object-aware filtering
- **Subject-aware masking using YOLO + rembg for accurate cinematic application**

**Recent Additions**:

### ✅ YOLO-Based Masking & Cinematic Grading (28/05/2025)

Used the `rembg` library and object detection principles to isolate subjects before applying the cinematic effect. This improves focus, blending, and visual clarity in complex scenes.

```python
# Install dependencies
!pip install rembg onnxruntime opencv-python matplotlib --quiet

# Upload image and preprocess
from google.colab import files
import numpy as np, cv2
from PIL import Image
from rembg import remove
import matplotlib.pyplot as plt

uploaded = files.upload()
input_image = Image.open(next(iter(uploaded))).convert("RGB")
input_np = np.array(input_image)

# Create soft mask
mask_np = remove(input_np, only_mask=True)
subject_mask = cv2.GaussianBlur(mask_np.astype(np.float32) / 255.0, (31, 31), 0)

# Apply orange-teal cinematic grading with vignette
def apply_cinematic_effect(img):
    img = img.astype(np.float32) / 255.0
    img[..., 0] *= 1.2  # Blue
    img[..., 1] *= 0.95 # Green
    img[..., 2] *= 1.05 # Red
    vignette = cv2.getGaussianKernel(img.shape[1], 250) @ cv2.getGaussianKernel(img.shape[0], 250).T
    vignette = np.dstack([vignette / vignette.max()] * 3)
    return (np.clip(img * vignette, 0, 1) * 255).astype(np.uint8)

# Blend subject-only grading
cinematic = apply_cinematic_effect(input_np)
blended = (cinematic * subject_mask[..., None] + input_np * (1 - subject_mask[..., None])).astype(np.uint8)
