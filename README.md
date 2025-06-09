# 🧠 Internship Project – AI-Powered Applications (May 2025)

This repository showcases the work I completed during my internship in May 2025. It includes four major AI/ML-based tasks: **Text-to-Image Generation**, **Sentiment Analysis using NLP**, **Spam Detection using SVM**, and **Image/Video Enhancement with Cinematic Filter**.

---

## 📅 Timeline

| Date        | Activity                                                                 |
|-------------|--------------------------------------------------------------------------|
| 12/05/2025  | Studied `NumPy`, `Pandas`, `Keras`, `TensorFlow`, `PyTorch`              |
| 13/05/2025  | Continued exploration of ML/DL libraries                                 |
| 14/05/2025  | ✅ Completed **Task 1**: Text-to-Image Generator using Stable Diffusion  |
| 15/05/2025  | ✅ Completed **Task 2**: Sentiment Analysis using NLP                    |
| 16/05/2025  | ✅ Completed **Task 3**: Email Spam Classification using SVM             |
| 17/05/2025  | 📌 Started **Task 4**: Researched Image/Video Enhancement techniques     |
| 18/05/2025  | ✅ Implemented cinematic filter for **Image Enhancement**                |
| 19/05/2025  | ✅ Extended functionality to **Video & Webcam** with `Streamlit` UI      |
| 20/05/2025  | ✅ Successfully enhanced **Video** with reduced pixel distortion         |
| 22/05/2025  | ✅ Completed **Real-time Enhancement** using **Webcam Feed**             |
| 23/05/2025  | ✅ Deployed Full **Cinematic Filter Studio** via **Streamlit Cloud**     |
| 24/05/2025  | 📌 Researched and fixed **frame capture issues** in deployed webcam mode |
| 25/05/2025  | ✅ Implemented **Hugging Face API** for predefined model-based enhancement |
| 26/05/2025  | 📌 Researched **image quality improvement** techniques post-enhancement |
| 27/05/2025  | ✅ Improved enhancement quality with **object-aware filtering**          |
| 28/05/2025  | ✅ Added **YOLO-based subject masking** for focused cinematic enhancement |
| 29/05/2025  | ✅ Successfully applied **YOLO masking** for object-focused image enhancement |
| 30/05/2025  | 📌 Researched **YOLO + masking** for object-aware **video enhancement** |
| 31/05/2025  | Started working on **video masking module**; achieved basic 20% filtering |
| 01–02/06/2025 | ✅ Researched and started video enhancement from scratch using `MoviePy` |
| 03–04/06/2025 | ✅ Developed working pipeline for **video+audio** processing and FPS/frame merging |
| 05/06/2025  | ✅ Implemented video file upload and basic processing UI |
| 06–07/06/2025 | ✅ Developed core cinematic enhancement using `MoviePy` and `OpenCV` |
| 08–09/06/2025 | ✅ Integrated full pipeline with letterbox + cinematic coloring + output video/audio |

---

## 🔧 Task 1: Text-to-Image Generation 🎨

**Description**:  
Created a web app using `Streamlit` to generate images from natural language prompts using **Stable Diffusion v2.1**.

**Model Used**: `stabilityai/stable-diffusion-2-1`  
**Libraries**: `Streamlit`, `Torch`, `Diffusers`, `PIL`, `GC`, `Platform`

**Features**:
- Interactive UI with adjustable parameters (image size, steps, guidance scale, seed)
- GPU/CPU optimized image generation
- Downloadable output
- Prompt & negative prompt support

---

## 💬 Task 2: Sentiment Analysis using NLP 🧾

**Description**:  
Built a sentiment classifier to label input text as **positive** or **negative** using `Naive Bayes`.

**Libraries Used**:  
`Scikit-learn`

**Process**:
- Collected 8 sample labeled texts
- Used `CountVectorizer` for feature extraction
- Trained `MultinomialNB` classifier
- Predicted sentiment of user-provided prompt

---

## 📧 Task 3: Spam Classification using SVM 🚫

**Description**:  
Developed a simple SVM-based email classifier to detect **Spam** vs **Not Spam**.

**Libraries Used**:  
`Scikit-learn`, `re` (Regex)

**Process**:
- Preprocessed emails using regex
- Used `CountVectorizer`
- Trained SVM with linear kernel
- Predicted new emails as spam or not

---

## 🎞️ Task 4: Cinematic Image/Video/Webcam Enhancement ✨

**Description**:  
Designed and implemented a **Cinematic Enhancement Filter** for images, videos, and real-time webcam feeds. Later extended to include **YOLO-based subject masking** and **MoviePy-based video enhancement** with audio syncing and cinematic filters.

**Libraries Used**:  
`OpenCV`, `Pillow`, `NumPy`, `Matplotlib`, `Google Colab`, `Streamlit`, `requests`, `Hugging Face API`, `rembg`, `onnxruntime`, `YOLO`, `MoviePy`

---

### ✅ Key Features Implemented

- Applies cinematic-style filter (contrast, tint, vignette, grain)
- Supports **image**, **video**, and **real-time webcam enhancement**
- UI built in `Streamlit` with **interactive sliders**
- ✅ Cloud deployment via **Streamlit Cloud**
- ✅ Used **Hugging Face pretrained model** via API for image enhancement (25/05/2025)
- ✅ YOLO-based **subject-aware masking** for cinematic blending (28–30/05/2025)
- ✅ Started `MoviePy`-based **video masking and enhancement** (31/05/2025 onward)
- ✅ Full video+audio processing implemented using `MoviePy` (03–09/06/2025)
- ✅ Brightness/contrast enhancement, red-blue tone boost, cinematic frame borders

---

### 📌 Enhancement Pipeline Functions

| Function Name              | Purpose                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| `CinematicFilter.apply()`  | Applies full cinematic pipeline                                         |
| `adjust_tint()`            | Adds film-like red-blue tint                                            |
| `add_vignette()`           | Adds dark borders for dramatic depth                                    |
| `add_film_grain()`         | Adds grain to simulate analog film                                      |
| `process_image_mode()`     | Applies cinematic filter to static image                                |
| `process_video_mode()`     | Enhances all video frames                                               |
| `process_webcam_mode()`    | Applies real-time cinematic filter via webcam                           |
| `streamlit cloud deploy`   | Deploys full app to **Streamlit Cloud**                                 |
| `huggingface_image_enhance()` | Uses Hugging Face API to apply pretrained enhancement model          |
| `object_aware_enhance()`   | Improves object clarity post-enhancement (27/05/2025)                   |
| `yolo_subject_mask()`      | Applies YOLO + rembg-based subject masking for image (28–29/05/2025)    |
| `yolo_video_mask_pipeline()` | 📌 In progress: YOLO + mask-based video enhancement (30/05/2025)       |
| `apply_cinematic_effect()` | ✅ MoviePy-based full pipeline: border, tint, FPS, audio sync (08/06/2025)|

---

## 📌 Technologies & Libraries Explored

| Category             | Tools / Libraries                                |
|----------------------|--------------------------------------------------|
| ML / DL Libraries    | `TensorFlow`, `Keras`, `PyTorch`                 |
| Data Handling        | `NumPy`, `Pandas`                                |
| Web App              | `Streamlit`, `Streamlit Cloud`                   |
| NLP & ML Models      | `Scikit-learn`, `Naive Bayes`, `SVM`             |
| Image Generation     | `Stable Diffusion`, `Diffusers`, `Torch`         |
| Enhancement Tools    | `OpenCV`, `Pillow`, `Matplotlib`, `Hugging Face API`, `rembg`, `YOLO`, `MoviePy`, `onnxruntime` |
| Deployment & Utilities | `GC`, `Platform`, `Google Colab`, `requests`    |

---

## 🚀 Summary

This internship helped me explore and implement:
- Real-world **text-to-image generation**
- NLP-based **sentiment classification**
- Email **spam filtering using SVM**
- Cinematic-style **media enhancement**
- Real-time **video & webcam filtering**
- ✅ Integrated **Hugging Face** models for enhancement
- ✅ YOLO-based **subject masking**
- ✅ Built full **MoviePy-based video enhancer** from scratch
- ✅ Audio+video sync, FPS handling, red-blue grading, frame enhancement

> ✅ Regular commits and progress updates were made between **12 May – 9 June 2025**.

---

**👨‍💻 Developed by:** Kunj Mori  
**📅 Internship Duration:** May 2025 - July 2025
