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
- Advanced settings with explanations
- Prompt & negative prompt support

**Main Functions Used**:
| Function Name           | Purpose                                                        |
|-------------------------|----------------------------------------------------------------|
| `load_model()`          | Loads and configures the Stable Diffusion pipeline             |
| `clear_memory()`        | Frees up GPU memory after image generation                     |
| `st.text_area()`        | Accepts user prompt and negative prompt                        |
| `pipe()`                | Generates the image from the prompt using diffusion model      |
| `st.download_button()`  | Allows downloading the generated image                         |

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

**Main Functions Used**:
| Function Name      | Purpose                                           |
|--------------------|---------------------------------------------------|
| `CountVectorizer()`| Converts text to numeric feature vectors          |
| `MultinomialNB()`  | Trains Naive Bayes classifier on labeled data     |
| `model.predict()`  | Predicts the sentiment of new user input          |

---

## 📧 Task 3: Spam Classification using SVM 🚫

**Description**:  
Developed a simple SVM-based email classifier to detect **Spam** vs **Not Spam**.

**Libraries Used**:  
`Scikit-learn`, `re` (Regex)

**Process**:
- Preprocessed emails using regex (lowercase, remove non-alphanumeric)
- Used `CountVectorizer` to vectorize the text
- Trained `SVC` with linear kernel
- Predicted category of user input email

**Main Functions Used**:
| Function Name           | Purpose                                                |
|-------------------------|--------------------------------------------------------|
| `preprocess()`          | Cleans text using regex                                |
| `CountVectorizer()`     | Converts emails to numeric format                      |
| `SVC(kernel='linear')`  | Trains a linear support vector machine                 |
| `model.predict()`       | Predicts spam or not spam based on user input          |

---

## 🎞️ Task 4: Image & Video Enhancement – Cinematic Filter ✨

**Description**:  
Designed and implemented a **Cinematic Enhancement Filter** for both images and videos. The filter enhances media by adjusting contrast, brightness, color, adding tint, vignette effect, and film grain. Integrated with a simple CLI and `Streamlit` UI.

**Libraries Used**:  
`OpenCV`, `Pillow`, `NumPy`, `Matplotlib`, `Google Colab`, `Streamlit`

**Key Features**:
- Applies cinematic-style enhancements to **images**, **videos**, and **real-time webcam feed**
- ✅ All modes combined and deployed in a unified **Streamlit app** on **Streamlit Cloud**
- Adjustable filter parameters for fine-tuning
- ✅ Enhanced video quality by reducing **pixel distortion artifacts**
- ✅ Implemented **real-time webcam enhancement** pipeline with `Streamlit`
- ✅ Researched and fixed **frame buffer issues** on cloud for webcam support
- Saves enhanced outputs and allows download

**Main Functions Used**:
| Function Name            | Purpose                                                   |
|--------------------------|-----------------------------------------------------------|
| `CinematicFilter.apply()`| Main enhancement pipeline (contrast, tint, vignette, etc.)|
| `adjust_tint()`          | Applies a red-blue cinematic tint                         |
| `add_vignette()`         | Adds dark corners for depth effect                        |
| `add_film_grain()`       | Introduces subtle grain to simulate film look             |
| `process_image_mode()`   | Applies filter to image and shows + downloads result      |
| `process_video_mode()`   | Processes all frames of a video and saves output          |
| `process_webcam_mode()`  | ✅ Streams **real-time webcam** with filter applied        |
| `streamlit cloud deploy` | ✅ Deploys the full enhancement app to the cloud           |

---

## 📌 Technologies & Libraries Explored

| Category             | Tools / Libraries                                |
|----------------------|--------------------------------------------------|
| ML / DL Libraries    | `TensorFlow`, `Keras`, `PyTorch`                 |
| Data Handling        | `NumPy`, `Pandas`                                |
| Web App              | `Streamlit`, `Streamlit Cloud`                   |
| NLP & ML Models      | `Scikit-learn`, `Naive Bayes`, `SVM`             |
| Image Generation     | `Stable Diffusion`, `Diffusers`, `Torch`         |
| Enhancement Tools    | `OpenCV`, `Pillow`, `Matplotlib`                 |
| Miscellaneous        | `Regex`, `GC`, `Platform`, `Google Colab`        |

---

## 🚀 Summary

This internship helped me explore advanced AI/ML concepts, including:
- Real-world **text-to-image synthesis**
- Building **NLP pipelines** from scratch
- Classifying textual data with **SVM**
- Applying **Cinematic Filters** to image and video data
- Enhancing **video quality** by reducing pixel distortion
- ✅ Achieved **real-time enhancement** of webcam feed using OpenCV + Streamlit
- ✅ Deployed full solution (image, video, webcam) to **Streamlit Cloud**
- 📌 Solved **frame capture issues** in real-time filtering post-deployment

> ✅ Regular commits and progress updates were made between 12–24 May 2025.

---

**👨‍💻 Developed by:** Kunj Mori  
**📅 Internship Duration:** May 2025 - July 2025
