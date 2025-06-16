# 🧠 DocuMind OCR App

DocuMind is an intelligent OCR-based Streamlit app that reads scanned documents, corrects typos, extracts keywords, generates concise summaries using NLP models, and translates summaries to English when needed.

> Built to showcase practical AI skills and impress Charpak reviewers with elegance, efficiency, and a little extra brain power. 🧼🧠✨

---

## 🚀 Features

- 📸 Upload scanned **images or PDFs**
- 🔍 Preprocess image for better OCR accuracy
- 🧽 Clean & format raw OCR text
- 🛠 Spellcheck and auto-correction
- 🧠 Extract top **keywords**
- 📚 Generate intelligent **summaries** using `facebook/bart-large-cnn`
- 🌐 Translate non-English summaries to **English**
- 🌍 Automatically detects language (using `langdetect`)

---

## 🧰 Tech Stack

| Layer         | Tool / Library                              |
| ------------- | -------------------------------------------- |
| Frontend      | Streamlit                                   |
| OCR Engine    | Tesseract OCR (`pytesseract`)                |
| NLP           | HuggingFace Transformers (`bart`, `opus-mt`) |
| Keyword Ext.  | `CountVectorizer` from Scikit-learn         |
| Summarization | `facebook/bart-large-cnn`                   |
| Translation   | `Helsinki-NLP/opus-mt-ROMANCE-en`           |
| Language Det. | `langdetect`                                |
| Preprocessing | OpenCV, PIL, regex                          |

---

## 📸 Screenshots
![Screenshot 2025-06-16 135346](https://github.com/user-attachments/assets/22d4c74a-1e43-47ee-88fc-8ae8c7abc687)

![image](https://github.com/user-attachments/assets/a9df50b4-4ab6-43c1-8b79-bc90bea0f255)


---

## 💡 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/ayulikhar/documind-ocr-app.git
   cd documind-ocr-app
