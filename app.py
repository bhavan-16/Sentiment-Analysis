import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./results"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

labels = ["Negative", "Neutral", "Positive"]

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🔍", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #ff416c, #ff4b2b, #6a11cb, #2575fc);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .glow-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #fff;
        background: rgba(0, 0, 0, 0.4);
        margin-top: 20px;
    }
    .positive { box-shadow: 0 0 25px #00ff88; }
    .neutral { box-shadow: 0 0 25px #ffaa00; }
    .negative { box-shadow: 0 0 25px #ff0033; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>🔍 Sentiment Analyzer</h1>", unsafe_allow_html=True)

user_input = st.text_area("Type your text here:")

if st.button("Analyze 🚀"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        pred_label = predictions.argmax(dim=1).cpu().item()
        confidence = predictions[0][pred_label].cpu().item() * 100
        sentiment = labels[pred_label]

        css_class = "positive" if sentiment == "Positive" else "neutral" if sentiment == "Neutral" else "negative"

        st.markdown(
            f"<div class='glow-card {css_class}'>"
            f"<p>Prediction:</p>"
            f"<h2>{sentiment}</h2>"
            f"<p>Confidence: {confidence:.2f}%</p>"
            f"</div>",
            unsafe_allow_html=True
        )
