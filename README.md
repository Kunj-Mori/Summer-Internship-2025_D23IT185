# 🧠 Internship Project – AI-Powered Applications (May 2025)

This repository showcases the work I completed during my internship in May 2025. It includes three major AI/ML-based tasks: **Text-to-Image Generation**, **Sentiment Analysis using NLP**, and **Spam Detection using SVM**.

---

## 📅 Timeline

| Date        | Activity                                                                 |
|-------------|--------------------------------------------------------------------------|
| 12/05/2025  | Studied `NumPy`, `Pandas`, `Keras`, `TensorFlow`, `PyTorch`              |
| 13/05/2025  | Continued exploration of ML/DL libraries                                 |
| 14/05/2025  | ✅ Completed **Task 1**: Text-to-Image Generator using Stable Diffusion  |
| 15/05/2025  | ✅ Completed **Task 2**: Sentiment Analysis using NLP                    |
| 16/05/2025  | ✅ Completed **Task 3**: Email Spam Classification using SVM             |

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
| Function Name     | Purpose                                                        |
|-------------------|----------------------------------------------------------------|
| `load_model()`    | Loads and configures the Stable Diffusion pipeline             |
| `clear_memory()`  | Frees up GPU memory after image generation                     |
| `st.text_area()`  | Accepts user prompt and negative prompt                        |
| `pipe()`          | Generates the image from the prompt using diffusion model      |
| `st.download_button()` | Allows downloading the generated image                    |

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
| Function Name      | Purpose                                                |
|--------------------|--------------------------------------------------------|
| `preprocess()`     | Cleans text using regex                                |
| `CountVectorizer()`| Converts emails to numeric format                      |
| `SVC(kernel='linear')` | Trains a linear support vector machine            |
| `model.predict()`  | Predicts spam or not spam based on user input          |

---

## 📌 Technologies & Libraries Explored

| Category             | Tools / Libraries                                |
|----------------------|--------------------------------------------------|
| ML / DL Libraries    | `TensorFlow`, `Keras`, `PyTorch`                 |
| Data Handling        | `NumPy`, `Pandas`                                |
| Web App              | `Streamlit`                                      |
| NLP & ML Models      | `Scikit-learn`, `Naive Bayes`, `SVM`             |
| Image Generation     | `Stable Diffusion`, `Diffusers`, `Torch`         |
| Miscellaneous        | `PIL`, `Regex`, `GC`, `Platform`                 |

---

## 🚀 Summary

This internship helped me explore advanced AI/ML concepts, including:
- Real-world **text-to-image synthesis**
- Building **NLP pipelines** from scratch
- Classifying textual data with **SVM**
- Using cutting-edge tools like **Stable Diffusion**

> ✅ Regular commits and progress updates were made between 12–16 May 2025.

---

**👨‍💻 Developed by:** Kunj Mori  
**📅 Internship Duration:** May 2025 
