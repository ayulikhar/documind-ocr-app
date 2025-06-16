# NLM prototype
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\Files\OCR_App\Tesseract_OCR\tesseract.exe"
import streamlit as st
from PIL import Image
import re
import numpy as np
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from transformers import pipeline
import fitz
from langdetect import detect
#import sentencepiece
from PyPDF2 import PdfReader

try:
    import cv2
except ImportError:
    print("⚠️ OpenCV not installed! Run `pip install opencv-python`.")
    exit()

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Initialize spell checker
spell = SpellChecker()

# OCR Cleaning Function
def clean_ocr_text(raw_text):
    text = re.sub(r'\n+', ' ', raw_text)            # Remove multiple line breaks
    text = re.sub(r'\|', '', text)                  # Remove pipe characters
    text = re.sub(r'-\s+', '', text)                # Join words broken with hyphen and space
    text = re.sub(r'\s{2,}', ' ', text)             # Remove extra spaces
    return text.strip()

# Typo Correction Function
def correct_spelling(text):
    tokens = word_tokenize(text)
    unknowns = spell.unknown(tokens)
    corrected = []
    for word in tokens:
        if word.lower() in unknowns:
            correction = spell.correction(word)
            corrected.append(correction if correction else word)
        else:
            corrected.append(word)
    return ' '.join(corrected)

def extract_text_from_pdf(pdf_file):
    # Read file as bytes
    pdf_bytes = pdf_file.read()
    text = ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        text = f"⚠️ Could not read PDF: {e}"
    return text

    
# Keyword Extraction
def extract_keywords(text, top_n=10):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    word_freq = np.asarray(X.sum(axis=0)).flatten()
    vocab = vectorizer.get_feature_names_out()
    keywords = sorted(zip(vocab, word_freq), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:top_n]]

# 💬 Load summarization model (this will download once & cache)
summarizer = pipeline("summarization",model="facebook/bart-large-cnn")
print(summarizer("Transformers are amazing tools for natural language processing tasks.")[0]['summary_text'])
#translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ROMANCE-en")  # for FR→EN, ES→EN, IT→EN etc.

# 📚 Generate summary from text

def generate_summary(text, max_length=120, min_length=30):
    try:
        # Clean weird whitespace
        text = text.strip().replace("\n", " ")
        
        if len(text.split()) < 20:
            return "⚠️ Text too short to summarize meaningfully."

        # Truncate to avoid index errors
        trimmed_text = text[:1000]  # adjust as needed
        summary = summarizer(trimmed_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    
    except IndexError:
        return "⚠️ Could not generate summary: text too short or model limit exceeded."
    
    except Exception as e:
        return f"⚠️ Could not generate summary: {e}"

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text() or ""
    return all_text

# 🔍 Preprocess image to boost OCR accuracy
def preprocess_image_for_ocr(pil_image):
    # Convert from PIL to numpy array
    np_image = np.array(pil_image)

    # Check if image has color channels
    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
        # Convert RGB to grayscale
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    else:
        # Already grayscale
        gray = np_image

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Convert back to PIL for pytesseract
    return Image.fromarray(thresh)


# 🧠 Full Pipeline
def documind_pipeline(image):
    # First OCR rough pass to detect language
    rough_text = pytesseract.image_to_string(image)

    # Detect language
    try:
        detected_lang = detect(rough_text)
        lang_map = {
            'en': 'eng',
            'fr': 'fra',
            'de': 'deu',
            'es': 'spa',
            'it': 'ita',
        }
        tess_lang = lang_map.get(detected_lang, 'eng')
    except:
        detected_lang = 'en'
        tess_lang = 'eng'

    # Run OCR with proper language
    raw_text = pytesseract.image_to_string(image, lang=tess_lang)
    cleaned = clean_ocr_text(raw_text)
    keywords = extract_keywords(cleaned)

    # ✨ Summary
    try:
        text_to_summarize = cleaned if len(cleaned) < 1000 else cleaned[:1000]
        summary_text = summarizer(cleaned, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
    except Exception as e:
        summary_text = f"⚠️ Could not generate summary: {e}"

    #Translate summary if not in English
    translated_summary = None
    #if detected_lang != 'en':
    #    try:
    #        translated_summary = translator(summary_text, max_length=100)[0]['translation_text']
    #    except Exception as e:
    #        translated_summary = f"Translation error: {e}"
    
    return {
        'raw_text': raw_text,
        'cleaned_text': cleaned,
        'summary': summary_text,
        'translated_summary': translated_summary,
        'keywords': keywords,
        'language_detected': tess_lang
    }

st.set_page_config(page_title="DocuMind OCR Cleaner 🧽", layout="wide")
st.title("🧠 DocuMind: OCR Text Cleaner and Analyzer")

uploaded_file = st.file_uploader("📤 Upload an image or PDF with text", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    result = None  # initialize to avoid name errors later

    if file_type.startswith("image/"):
        image = Image.open(uploaded_file)
        preprocessed_image = preprocess_image_for_ocr(image)
        with st.spinner("🔍 Analyzing image..."):
            result = documind_pipeline(preprocessed_image)

    elif file_type == "application/pdf":
        with st.spinner("🔍 Analyzing PDF..."):
            try:
                extracted_text = extract_text_from_pdf(uploaded_file)
                cleaned = clean_ocr_text(extracted_text)
                corrected = correct_spelling(cleaned)
                keywords = extract_keywords(corrected)
                result = {
                    'raw_text': extracted_text,
                    'cleaned_text': corrected,
                    'summary': "Click the button below to generate summary.",
                    'translated_summary': None,
                    'keywords': keywords,
                    'language_detected': detect(corrected)
                }
            except Exception as e:
                st.error(f"PDF Error: {e}")

    if result:
        st.info(f"🌍 Detected OCR Language: `{result['language_detected']}`")
        st.text_area("🧼 Cleaned Text", result['cleaned_text'], height=200)
        st.success("🔑 Keywords: " + ", ".join(result['keywords']))

        if st.button("✨ Generate Summary"):
            try:
                summary_text = generate_summary(result['cleaned_text'])
                st.subheader("📚 Summary")
                st.success(summary_text)
            except Exception as e:
                st.error(f"Could not generate summary: {e}")


    # Show summaries if available in result
    if result.get('summary') and "Click the button" not in result['summary']:
        st.subheader("📚 Auto-generated Summary")
        st.success(result['summary'])
        
    if result.get('translated_summary'):
        st.subheader("🌐 Translated Summary (English)")
        st.info(result['translated_summary'])

    