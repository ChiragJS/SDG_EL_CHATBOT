import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

CSV_FILE = 'dataset.csv' 

@st.cache_data
def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['combined_info'] = df.apply(lambda row: ' '.join(row.dropna().astype(str).tolist()), axis=1)
        return df
    except FileNotFoundError:
        st.error(f"Error: CSV file '{file_path}' not found.")
        return pd.DataFrame()

@st.cache_resource
def get_vectorizer_and_matrix(df):
    if df.empty:
        return None, None
    vectorizer = TfidfVectorizer().fit(df['combined_info'])
    tfidf_matrix = vectorizer.transform(df['combined_info'])
    return vectorizer, tfidf_matrix

@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel('gemini-1.5-flash')

df_aqi = load_csv_data(CSV_FILE)
vectorizer, tfidf_matrix = get_vectorizer_and_matrix(df_aqi)
model = get_gemini_model()

def get_relevant_csv_info(query, df, vectorizer, tfidf_matrix, top_n=3):
    if df.empty or vectorizer is None or tfidf_matrix is None:
        return ""

    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[-top_n:][::-1]

    relevant_info = []
    for i in related_docs_indices:
        if cosine_similarities[i] > 0.1:
            relevant_info.append(df.iloc[i].to_dict())

    if relevant_info:
        formatted_info = "\n\nRelevant AQI Information:\n"
        for item in relevant_info:
            formatted_info += f"- AQI: {item.get('AQI', 'N/A')}\n"
            formatted_info += f"  percentCO2: {item.get('percentCO2', 'N/A')}%\n"
            formatted_info += f"  percentN2: {item.get('percentN2', 'N/A')}%\n"
            formatted_info += f"  percentO2: {item.get('percentO2', 'N/A')}%\n"
            formatted_info += f"  percentOthers: {item.get('percentOthers', 'N/A')}%\n"
            formatted_info += f"  ppm_Alcohol: {item.get('ppm_Alcohol', 'N/A')} ppm\n"
            formatted_info += f"  ppm_CO: {item.get('ppm_CO', 'N/A')} ppm\n"
            formatted_info += f"  ppm_CO2: {item.get('ppm_CO2', 'N/A')} ppm\n"
            formatted_info += f"  ppm_NH3: {item.get('ppm_NH3', 'N/A')} ppm\n"
            formatted_info += f"  ppm_NOx: {item.get('ppm_NOx', 'N/A')} ppm\n"
            formatted_info += f"  ppm_VOC: {item.get('ppm_VOC', 'N/A')} ppm\n"
            formatted_info += f"  ratio: {item.get('ratio', 'N/A')}\n"
            formatted_info += f"  rawADC: {item.get('rawADC', 'N/A')}\n"
            formatted_info += "----\n"
        return formatted_info
    return ""

def get_gemini_response(prompt_with_context, chat_history):
    try:
        history_for_gemini = []
        for msg in chat_history:
            if msg["role"] == "user":
                history_for_gemini.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                history_for_gemini.append({"role": "model", "parts": [msg["content"]]})

        chat = model.start_chat(history=history_for_gemini)
        response = chat.send_message(prompt_with_context)
        return response.text
    except Exception as e:
        st.error(f"Error communicating with Gemini: {e}")
        return "I apologize, but I'm having trouble connecting to the AI at the moment."

st.set_page_config(page_title="AQI Chatbot", layout="centered")
st.title("🌫️ AQI Info Chatbot")
st.markdown("""
<style>
    section.main[data-testid="stSidebar"] {display: none !important;}
    .stChatInput, .stTextInput input {
        font-size: 16px;
        padding: 12px;
        border-radius: 10px;
    }
    .stTextInput input:focus {
        border: 2px solid #4CAF50;
    }
    .stButton button {
        background-color: transparent;
        color: red;
        border: 2px solid red;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        transition: box-shadow 0.3s ease;
    }
    .stButton button:hover {
        box-shadow: 0 0 10px red;
    }
    .main > div {
        padding-bottom: 100px;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message and press Enter...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant_csv_data = get_relevant_csv_info(user_input, df_aqi, vectorizer, tfidf_matrix)
            full_prompt = (
                f"You are a helpful assistant that answers questions based on the provided AQI data. "
                f"Do not make up information.\n\n{relevant_csv_data}\n\nUser query: {user_input}"
            )
            response_text = get_gemini_response(full_prompt, st.session_state.messages)
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
