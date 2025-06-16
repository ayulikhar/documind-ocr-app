# ocr_cleaner_app.py
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


# 🔧 Initialize spell checker
spell = SpellChecker()

# 🧹 OCR Cleaning Function
def clean_ocr_text(raw_text):
    text = re.sub(r'\n+', ' ', raw_text)            # Remove multiple line breaks
    text = re.sub(r'\|', '', text)                  # Remove pipe characters
    text = re.sub(r'-\s+', '', text)                # Join words broken with hyphen and space
    text = re.sub(r'\s{2,}', ' ', text)             # Remove extra spaces
    return text.strip()

# 🩹 Typo Correction Function
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

# 🧲 Keyword Extraction
def extract_keywords(text, top_n=10):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    word_freq = np.asarray(X.sum(axis=0)).flatten()
    vocab = vectorizer.get_feature_names_out()
    keywords = sorted(zip(vocab, word_freq), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:top_n]]

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
    raw_text = pytesseract.image_to_string(image)
    print("🧾 OCR Raw Text:", raw_text)
    cleaned = clean_ocr_text(raw_text)
    corrected = correct_spelling(cleaned)
    keywords = extract_keywords(corrected)

    return {
        'raw_text': raw_text,
        'cleaned_text': cleaned,
        'corrected_text': corrected,
        'keywords': keywords
    }

# 🖼️ Streamlit UI
st.set_page_config(page_title="DocuMind OCR Cleaner 🧽", layout="wide")
st.title("🧠 DocuMind: OCR Text Cleaner and Analyzer")

uploaded_file = st.file_uploader("📤 Upload an image with text", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #st.image(image, caption="Uploaded Image", use_column_width=True)

    # 💡 Preprocess for OCR
    preprocessed_image = preprocess_image_for_ocr(image)
    #st.image(preprocessed_image, caption="Preprocessed Image for OCR", use_column_width=True)

    st.markdown("---")
    with st.spinner("🔍 Extracting and analyzing text..."):
        preprocessed_image = preprocess_image_for_ocr(image)
        result = documind_pipeline(preprocessed_image)


    #st.subheader("📄 Raw OCR Text")
    #st.code(result['raw_text'])

    st.subheader("🧼 Cleaned Text")
    #st.code(result['cleaned_text'])
    st.text_area("Cleaned Text", result['cleaned_text'], height=200)

    #st.subheader("✅ Typo-Corrected Text")
    #st.code(result['corrected_text'])
    #st.text_area("Typo-Corrected Text", result['corrected_text'], height=200)

    st.subheader("🔑 Extracted Keywords")
    st.success(", ".join(result['keywords']))

    st.subheader("🧪 Debug Output")
    st.text(result['raw_text'] or "⚠️ No text extracted!")


