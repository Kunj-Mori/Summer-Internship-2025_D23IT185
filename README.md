## 📅 Timeline (Weekly Tree View)

<details>
<summary><strong>📦 Week 1 (12 May – 18 May)</strong></summary>

| Date       | Day        | Activity                                                                 |
|------------|------------|--------------------------------------------------------------------------|
| 12/05/2025 | Monday     | Studied `NumPy`, `Pandas`, `Keras`, `TensorFlow`, `PyTorch`              |
| 13/05/2025 | Tuesday    | Continued exploration of ML/DL libraries                                 |
| 14/05/2025 | Wednesday  | ✅ Completed **Task 1**: Text-to-Image Generator using Stable Diffusion  |
| 15/05/2025 | Thursday   | ✅ Completed **Task 2**: Sentiment Analysis using NLP                    |
| 16/05/2025 | Friday     | ✅ Completed **Task 3**: Email Spam Classification using SVM             |
| 17/05/2025 | Saturday   | 📌 Started **Task 4**: Researched Image/Video Enhancement techniques     |
| 18/05/2025 | Sunday     | ✅ Implemented cinematic filter for **Image Enhancement**                |

</details>

<details>
<summary><strong>📦 Week 2 (19 May – 25 May)</strong></summary>

| Date       | Day        | Activity                                                                 |
|------------|------------|--------------------------------------------------------------------------|
| 19/05/2025 | Monday     | ✅ Extended functionality to **Video & Webcam** with `Streamlit` UI      |
| 20/05/2025 | Tuesday    | ✅ Successfully enhanced **Video** with reduced pixel distortion         |
| 22/05/2025 | Thursday   | ✅ Completed **Real-time Enhancement** using **Webcam Feed**             |
| 23/05/2025 | Friday     | ✅ Deployed Full **Cinematic Filter Studio** via **Streamlit Cloud**     |
| 24/05/2025 | Saturday   | 📌 Researched and fixed **frame capture issues** in deployed webcam mode |
| 25/05/2025 | Sunday     | ✅ Implemented **Hugging Face API** for predefined model-based enhancement |

</details>

<details>
<summary><strong>📦 Week 3 (26 May – 01 June)</strong></summary>

| Date       | Day        | Activity                                                                 |
|------------|------------|--------------------------------------------------------------------------|
| 26/05/2025 | Monday     | 📌 Researched **image quality improvement** techniques post-enhancement |
| 27/05/2025 | Tuesday    | ✅ Improved enhancement quality with **object-aware filtering**          |
| 28/05/2025 | Wednesday  | ✅ Added **YOLO-based subject masking** for focused cinematic enhancement |
| 29/05/2025 | Thursday   | ✅ Applied **YOLO masking** successfully on images                       |
| 30/05/2025 | Friday     | 📌 Researched **YOLO + masking** for **video enhancement**              |
| 31/05/2025 | Saturday   | Started working on **video masking module**; achieved 20% filtering     |

</details>

<details>
<summary><strong>📦 Week 4 (02 June – 08 June)</strong></summary>

| Date       | Day        | Activity                                                                 |
|------------|------------|--------------------------------------------------------------------------|
| 01–02/06   | Sun–Mon    | ✅ Researched and restarted video enhancement using `MoviePy` from scratch |
| 03–04/06   | Tue–Wed    | ✅ Developed working pipeline for **video+audio** processing, FPS/frame merging |
| 05/06/2025 | Thursday   | ✅ Started coding file upload + processing logic                         |
| 06–07/06   | Fri–Sat    | ✅ Developed **cinematic effect functions** using MoviePy + OpenCV        |
| 08/06/2025 | Sunday     | ✅ Finalized `apply_cinematic_effect()` with border, color grading, and output |

</details>

<details>
<summary><strong>📦 Week 5 (09 June – 15 June)</strong></summary>

| Date       | Day        | Activity                                                                 |
|------------|------------|--------------------------------------------------------------------------|
| 10/06/2025 | Monday     | ✅ Fine-tuned **YOLO-based video enhancement** pipeline                  |
| 11/06/2025 | Tuesday    | ✅ Tested on **human-focused videos**, improved clarity and accuracy     |
| 12/06/2025 | Wednesday  | ✅ Fixed **video enhancement stability and transitions**                 |
| 13/06/2025 | Thursday   | ✅ Final improvements in object-aware filtering for **video**            |
| 14/06/2025 | Friday     | ✅ Fine-tuned model post-processing for enhanced results                 |
| 15/06/2025 | Saturday   | ✅ Finalized detection model + filter combination                        |

</details>

<details>
<summary><strong>📦 Week 6 (16 June – 22 June)</strong></summary>

| Date       | Day        | Activity                                                                 |
|------------|------------|--------------------------------------------------------------------------|
| 16/06/2025 | Sunday     | 🔍 Started researching **webcam-based real-time cinematic enhancement** |
| 17/06/2025 | Monday     | 🔍 Continued segmentation and real-time video background studies         |
| 18/06/2025 | Tuesday    | ✅ Started development of **MediaPipe-based segmentation** model         |
| 19/06/2025 | Wednesday  | ✅ Local implementation with **virtualenv + OpenCV**                     |
| 20/06/2025 | Thursday   | ✅ Real-time filter with **MediaPipe + vignette + blur + color grading** |
| 21/06/2025 | Friday     | 🔄 Testing integration with webcam feed (real-time pipeline)             |
| 22/06/2025 | Saturday   | ✅ Full setup tested across different lighting conditions                |

</details>

## 🔧 Task 1: Text-to-Image Generation 🎨

**Description**:  
Created a web app using `Streamlit` to generate images from natural language prompts using **Stable Diffusion v2.1**.

**Model Used**:  
`stabilityai/stable-diffusion-2-1`

**Libraries**:  
`Streamlit`, `Torch`, `Diffusers`, `PIL`, `GC`, `Platform`

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
Designed and implemented a **Cinematic Enhancement Filter** for images, videos, and real-time webcam feeds. Extended later to include:
- **YOLO-based subject masking**
- **MoviePy-based video enhancement**
- **Real-time MediaPipe-based webcam enhancement**

**Libraries Used**:  
`OpenCV`, `Pillow`, `NumPy`, `Matplotlib`, `Google Colab`, `Streamlit`, `requests`,  
`Hugging Face API`, `rembg`, `onnxruntime`, `YOLO`, `MoviePy`, `MediaPipe`

---

### ✅ Key Features Implemented

- Applies cinematic-style filter (contrast, tint, vignette, grain)
- Supports **image**, **video**, and **real-time webcam enhancement**
- UI built in `Streamlit` with **interactive sliders**
- ✅ Cloud deployment via **Streamlit Cloud**
- ✅ Used Hugging Face pretrained model via API for image enhancement (25/05/2025)
- ✅ YOLO-based **subject-aware masking** for cinematic blending (28–30/05/2025)
- ✅ MoviePy-based **video masking + enhancement** (31/05–09/06/2025)
- ✅ MediaPipe-based **webcam cinematic enhancement** (16/06–20/06/2025)

---

### 📌 Enhancement Pipeline Functions

| Function Name                 | Purpose                                                                 |
|------------------------------|-------------------------------------------------------------------------|
| `CinematicFilter.apply()`     | Applies full cinematic pipeline                                         |
| `adjust_tint()`               | Adds film-like red-blue tint                                            |
| `add_vignette()`              | Adds dark borders for dramatic depth                                    |
| `add_film_grain()`            | Adds grain to simulate analog film                                      |
| `process_image_mode()`        | Applies cinematic filter to static image                                |
| `process_video_mode()`        | Enhances all video frames                                               |
| `process_webcam_mode()`       | Applies real-time cinematic filter via webcam                           |
| `streamlit cloud deploy`      | Deploys full app to Streamlit Cloud                                     |
| `huggingface_image_enhance()` | Uses Hugging Face API to apply pretrained enhancement model             |
| `object_aware_enhance()`      | Improves object clarity post-enhancement (27/05/2025)                   |
| `yolo_subject_mask()`         | YOLO + rembg-based subject masking (28–29/05/2025)                      |
| `yolo_video_mask_pipeline()`  | In-progress: YOLO + mask-based video enhancement (30/05/2025)           |
| `apply_cinematic_effect()`    | Full MoviePy video filter (FPS, audio sync, red-blue tint) (08/06/2025)|
| `mediapipe_webcam_enhance()`  | MediaPipe-based real-time enhancement + background blur (20/06/2025)    |

---

## 📌 Technologies & Libraries Explored

| Category             | Tools / Libraries                                |
|----------------------|--------------------------------------------------|
| ML / DL Libraries    | `TensorFlow`, `Keras`, `PyTorch`                 |
| Data Handling        | `NumPy`, `Pandas`                                |
| Web App              | `Streamlit`, `Streamlit Cloud`                   |
| NLP & ML Models      | `Scikit-learn`, `Naive Bayes`, `SVM`             |
| Image Generation     | `Stable Diffusion`, `Diffusers`, `Torch`         |
| Enhancement Tools    | `OpenCV`, `Pillow`, `Matplotlib`, `Hugging Face API`, `rembg`, `YOLO`, `MoviePy`, `onnxruntime`, `MediaPipe` |
| Deployment & Utilities | `GC`, `Platform`, `Google Colab`, `requests`    |

---

## 🚀 Summary

This internship helped me explore and implement:
- Real-world **text-to-image generation**
- NLP-based **sentiment classification**
- Email **spam filtering using SVM**
- Cinematic-style **media enhancement**
- Real-time **video & webcam filtering**
- ✅ Integrated Hugging Face models for enhancement
- ✅ YOLO-based subject masking
- ✅ Full MoviePy-based video enhancer from scratch
- ✅ Audio+video sync, FPS handling, red-blue grading, frame enhancement
- ✅ MediaPipe-based **real-time webcam enhancement with blur, vignette, and tint**

> ✅ Regular commits and progress updates were made between **12 May – 20 June 2025**

---

**👨‍💻 Developed by:** Kunj Mori  
**📅 Internship Duration:** May 2025 - July 2025
