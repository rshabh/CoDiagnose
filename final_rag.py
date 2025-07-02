import base64
import io
import os
import json
import re
import sqlite3
import textwrap
from typing import List
import streamlit as st
from dotenv import load_dotenv
import torch
import pandas as pd
from fpdf import FPDF
import numpy as np
import pickle
import joblib
import ast

from opentelemetry.semconv._incubating.attributes.db_attributes import DB_NAME
from thefuzz import process
from PIL import Image
import urllib.parse

# Import LangChain modules and additional utilities for chaining.
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------
# ReportLab-based Medical Report Generator (for Feature 1)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Frame
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment, FileContent, FileName, FileType, Disposition

# --------------------------
# Page Configuration.
st.set_page_config(page_title="CoDiagnose: Healthcare Assistant (Using PubMedBERT & OpenAI via Chroma)",
                   page_icon="🩺", layout="wide")


# --------------------------
# Dummy translation function.
def t(text):
    return text

# --------------------------
# Load environment variables and set OpenAI credentials.
load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    st.error("OpenAI API key not found in environment variables.")

# --------------------------
# Custom CSS styling.
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

        # body {
        #     font-family: 'DM Sans', sans-serif;
        #     background-color: #0d1b2a;
        #     color: #f0e9dc;
        # }
        # 
        # .title {
        #     font-size: 50px;
        #     font-weight: 600;
        #     color: #49bcde;  /* warm orange */
        #     text-align: center;
        #     margin-bottom: 10px;
        #     text-shadow: 0px 0px 10px rgba(255, 140, 66, 0.5); /* orange glow */
        # }
        # 
        # .subtitle {
        #     font-size: 24px;
        #     color: #d9cfc3;  /* light beige-gray */
        #     text-align: center;
        #     font-weight: 300;
        #     margin-bottom: 40px;
        # }
        # 
        # .section {
        #     background: #1e1e1e;
        #     padding: 25px;
        #     border-radius: 15px;
        #     box-shadow: 0px 4px 12px rgba(255, 140, 66, 0.15);  /* soft orange */
        #     margin-bottom: 20px;
        #     transition: all 0.3s ease-in-out;
        # }

        # .section:hover {
        #     transform: translateY(-5px);
        #     box-shadow: 0px 6px 16px rgba(255, 140, 66, 0.3); /* brighter orange glow on hover */
        # }

        .sidebar-text {
            font-size: 3rem;
            color: #1c1913 ;
            text-align: center;
            margin-bottom: 20px;
        }

        .contact {
            text-align: left;
            font-size: 20px;
            color: #f0e9dc;
            margin-top: 40px;
        }

        .contact a {
            color: #ff8c42;
            text-decoration: none;
        }

        .profile-links {
            text-align: center;
            margin-top: 20px;
        }

        .profile-links a {
            font-size: 18px;
            color: #ffa94d;
            text-decoration: none;
            margin: 0 15px;
            font-weight: 600;
        }

        [data-testid="stImageContainer"] img {
    border-radius: 20px; /* curved sides */
    height: fit-content;
    width: 100%;
    # margin: 0 auto;
    # display: block;
    transition: transform 0.3s ease;
}

[data-testid="stImageContainer"]:hover img {
    
}

[data-testid="stSidebarContent"] {
    background-color: #7cc4db  !important;  /* Dark greyish background */
    padding: 20px;
    box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
}

[data-testid="stSidebarContent"] * {
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebarContent"] .css-1v3fvcr {
    background-color: transparent !important;
}
  body {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f5f5f7;
    color: #1c1c1e;
    margin: 0; padding: 0;
  }
  .title {
    font-size: 5rem; font-weight: 600; color: #000;
    text-align: center; margin: 2rem 0 1rem;
    color: #49bcde;
    text-shadow: 0px 0px 10px rgba(255, 140, 66, 0.5);
  }
  .subtitle {
    font-size: 1.25rem; color: #3c3c4399;
    text-align: center; font-weight: 400; margin-bottom: 2.5rem;
  }
  .section {
    background: #fff; padding: 2.5rem;
    border-radius: 1rem; 
    margin-bottom: 3rem; transition: transform .3s, box-shadow .3s;
  }
  .section + .section { margin-top: 3rem; }
  [data-testid="stTextInput"] .stTextInput>div>input {
    width:100%; padding:.75rem 1rem;
    border:1px solid #d1d1d6; border-radius:.75rem;
    background:#fafafa; font-size:1rem;
    transition:border-color .3s, box-shadow .3s;
  }
 
}
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Main Title & Subtitle.
st.markdown("<div class='title'>👩🏼‍⚕️CoDiagnose: Easy Assistance For Better Care🩺</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ease of work between Doctor & Patient</div>",
            unsafe_allow_html=True)
st.image("utils/app_profile.png", use_container_width=True)

# --------------------------
# --- Feature 1: Healthcare Prescription Insight (RAG-based) ---
# Enhanced Medical Prompt Template for RAG.
MEDICAL_ENHANCED_PROMPT_TEMPLATE = """
Below is aggregated information extracted from trusted medical sources (e.g., NIH, PubMed, BioASQ, etc.):
{aggregated_answers}

You are a seasoned medical expert with over 20 years of experience in clinical practice, patient care, and research. Using only the aggregated information extracted from trusted medical sources (e.g., National Institutes of Health, PubMed, BioASQ), answer the following query with extreme detail, reliability, and specificity. Your response should be tailored for both patients and healthcare professionals.

Now, generate a comprehensive and medically sound response with the following structure:

1. **Identified Medical Issue:**
   - Clearly state the specific disease or condition relevant to the patient's symptoms.

2. **Medical Interpretation:**
   - Provide a precise interpretation of the condition based on the aggregated data.

3. **Local Translations:**
   - Translate the name of the disease/condition into at least five Indian languages, such as Hindi, Bengali, Telugu, Marathi, and Tamil.

4. **Immediate Care Suggestions:**
   - Offer practical advice for immediate care based on the current situation.

5. **Disease Stages and Severity:**
   - Describe the different stages of the disease along with their severity ranges.

6. **Treatment Options:**
   - Provide comprehensive treatment recommendations, including:
     - **Pharmacological Treatments:** Detail chemical treatments with specific drug names, active ingredients, and dosage information.
     - **Non-Pharmacological Treatments:** Include organic or alternative treatment options.
     - **Application Guidelines:** Offer guidance on when and how to apply each treatment option.

7. **Recommended Medications:**
   - List specific medications pertinent to the patient's condition, including:
     - **Drug Name:** The generic and, if necessary, brand name.
     - **Dosage:** Recommended dosage and frequency.
     - **Administration:** Instructions on how to take the medication.
     - **Precautions:** Any important precautions or potential side effects.

8. **Additional Insights:**
   - Include any other important observations, recommendations, or considerations relevant to the patient's condition.

**Instructions:**
- Base your recommendations strictly on the provided data.
- Tailor your answer to the specific patient scenario indicated.

**Aggregated Information:**
[Insert aggregated information from trusted medical sources here.]

**Query:**
"{user_query}"
"""

# Define a new prompt template for disease recommendations.
DISEASE_RECOMMENDATION_PROMPT_TEMPLATE = """
You are a seasoned medical expert with over 20 years of experience in clinical practice, patient care, and research.
Based on the following predicted diseases:
{predicted_diseases}

Provide a detailed, evidence-based overview for each disease. For each disease, include:
1. **Brief Overview:** A short description of the disease.
2. **Common Symptoms:** The common symptoms associated with the disease.
3. **Biological Terminology:** The scientific or biological terms for the disease.
4. **Diagnostic Recommendations:** The best available diagnostic tests and procedures.
5. **Disease Severity and Staging:** A description of the potential stages and severity.
6. **Important Cautions:** Critical cautions and urgent actions required.
7. **Additional Insights:** Other important observations or recommendations.

Ensure the response is clear, precise, and tailored for both patients and healthcare professionals.

**Query:**
"{query}"
"""

# Define paths.
CHROMA_INDEX_PATH = "chroma_index"
DATA_PATH = "data/"


# --------------------------
# Function to load the embedding model.
def load_embedding_model() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using device: {device}")
    return HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings", model_kwargs={"device": device})


# --------------------------
# Function to load JSON documents and create Document objects.
@st.cache_data(show_spinner=True)
def load_documents() -> List[Document]:
    documents = []
    filenames = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".json")]
    for file_path in filenames:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "questions" in data:
                    for question in data["questions"]:
                        body = question.get("body", "").strip()
                        ideal_answers = question.get("ideal_answer", [])
                        exact_answers = question.get("exact_answer", [])
                        snippets = question.get("snippets", [])
                        snippet_texts = " ".join([s.get("text", "").strip() for s in snippets])
                        combined_context = f"Question: {body}\n"
                        if ideal_answers:
                            combined_context += f"Ideal Answer(s): {'; '.join(ideal_answers)}\n"
                        if exact_answers:
                            flat_exact = [item for sublist in exact_answers for item in sublist] \
                                if all(isinstance(item, list) for item in exact_answers) else exact_answers
                            combined_context += f"Exact Answer(s): {'; '.join(flat_exact)}\n"
                        if snippet_texts:
                            combined_context += f"Snippets: {snippet_texts}"
                        documents.append(Document(page_content=combined_context,
                                                  metadata={"qid": question.get("id", ""),
                                                            "source": os.path.basename(file_path)}))
                elif isinstance(data, list):
                    for entry in data:
                        context = entry.get("context", "").strip()
                        if context:
                            documents.append(
                                Document(page_content=context, metadata={"source": os.path.basename(file_path)}))
                elif isinstance(data, dict):
                    for key, entry in data.items():
                        contexts = entry.get("CONTEXTS", "")
                        context = "\n\n".join(contexts) if isinstance(contexts, list) else str(contexts)
                        context = context.strip()
                        if context:
                            documents.append(Document(page_content=context,
                                                      metadata={"source": os.path.basename(file_path), "qid": key}))
                else:
                    st.error(f"Unrecognized JSON structure in {file_path}")
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    st.info(f"Total documents loaded: {len(documents)}")
    return documents


# --------------------------
# Function to create and persist the Chroma index.
@st.cache_resource(show_spinner=True)
def create_chroma_index():
    documents = load_documents()
    if not documents:
        st.error("No documents found in data folder.")
        return None
    embedding_model = load_embedding_model()
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=CHROMA_INDEX_PATH)
    vectorstore.persist()
    st.success(f"Chroma index created and persisted at '{CHROMA_INDEX_PATH}'.")
    return vectorstore


# --------------------------
# Function to load the Chroma index.
@st.cache_resource(show_spinner=True)
def load_chroma_index():
    embedding_model = load_embedding_model()
    try:
        vectorstore = Chroma(persist_directory=CHROMA_INDEX_PATH, embedding_function=embedding_model)
        st.success("Chroma index loaded successfully.")
        return vectorstore
    except Exception as e:
        st.error(f"Error loading Chroma index: {e}")
        return None


# --------------------------
# Retrieve top-k relevant documents using Chroma.
def retrieve_docs(query, k=5):
    vectorstore = load_chroma_index()
    if vectorstore is None:
        st.error("Chroma index not loaded.")
        return []
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return results


# --------------------------
# Preprocess text for keyword filtering.
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------
# Reciprocal Rank Fusion (RRF) to re-rank retrieval results.
def reciprocal_rank_fusion(results: List[List[Document]], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = sorted(
        [(doc, score) for doc, score in fused_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    return [Document(page_content=doc) for doc, _ in reranked_results]

# --------------------------
# Process a query and generate a RAG-based prescription/insight using RRF.
def process_query(query_text, predicted_diseases="", custom_template=None):
    embedding_function = load_embedding_model()
    try:
        db = Chroma(persist_directory="chroma_index", embedding_function=embedding_function)
    except Exception as e:
        st.error("Error loading the Chroma vector store. Please update the database first.")
        return ""
    basic_results = db.similarity_search_with_relevance_scores(query_text, k=15)
    st.write(f"Retrieved {len(basic_results)} documents from Chroma DB (basic retrieval).")

    prompt_chain = ChatPromptTemplate(
        input_variables=["question"],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=[],
                    template="You are a human health expert specialized in biomedical research and clinical practice."
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["question"],
                    template="Generate 4 specific search queries related to: {question}\nOUTPUT (each on a new line):"
                )
            )
        ]
    )
    generate_queries = (
            prompt_chain
            | ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o", temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
    )
    ragfusion_chain = generate_queries | db.as_retriever().map() | reciprocal_rank_fusion
    aggregated_answers = "\n\n---\n\n".join([
        doc.page_content if hasattr(doc, "page_content")
        else doc[0].page_content if hasattr(doc[0], "page_content")
        else doc if isinstance(doc, str)
        else str(doc)
        for doc in ragfusion_chain
    ])
    source = t("Based upon trusted medical sources (NIH, PubMed, BioASQ, etc.).")

    # Determine prompt text based on available options.
    if custom_template:
        prompt_text = custom_template
    elif predicted_diseases:
        prompt_text = DISEASE_RECOMMENDATION_PROMPT_TEMPLATE.format(
            predicted_diseases=predicted_diseases,
            query=query_text
        )
    else:
        prompt_text = MEDICAL_ENHANCED_PROMPT_TEMPLATE.format(
            aggregated_answers=aggregated_answers,
            user_query=query_text
        )

    # Get model response.
    model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    response = model.predict(prompt_text)
    response_text = response.strip()

    # If a custom template was used, try to extract the patient information block.
    patient_info = ""
    if custom_template:
        lines = custom_template.splitlines()
        capture = False
        for line in lines:
            if "--- Patient Information ---" in line:
                capture = True
                continue
            if capture and line.strip() == "":
                # Stop capturing when an empty line is found (or adjust as needed)
                break
            if capture:
                patient_info += line.strip() + "\n"
        if patient_info:
            patient_info = "### Patient Information:\n" + patient_info.strip() + "\n\n"

    formatted_response = f"""
{patient_info}**Source:** {source}

**Response from the RAG model:**  
{response_text}
"""
    # For Feature 1, persist the raw response in a dedicated key.
    st.session_state["last_insight"] = response_text
    return formatted_response.strip()


def generate_medical_report(name, age, gender, response_text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 50
    line_height = 15
    max_chars_per_line = 100  # Approx width to prevent overflow

    def draw_header(first_page=False):
        if first_page:
            c.setFillColor(colors.HexColor("#003366"))
            c.setFont("Helvetica-Bold", 24)
            c.drawString(margin, height - 60, "CoDiagnose")

            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin, height - 100, "Patient Medical Report")

    def draw_footer():
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(colors.grey)
        c.drawCentredString(width / 2, 40, "Generated by Co Diagnose")

    # ---- Start Page ----
    draw_header(first_page=True)

    # Personal Info
    y = height - 140
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    c.drawString(margin, y, f"Name: {name}")
    c.drawString(margin, y - 20, f"Age: {age}")
    c.drawString(margin, y - 40, f"Gender: {gender}")

    y -= 80
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Healthcare Insights:")
    y -= 20

    # Split and wrap lines
    c.setFont("Helvetica", 12)
    lines = response_text.splitlines()
    first_page = True
    for line in lines:
        wrapped_lines = textwrap.wrap(line, width=max_chars_per_line)
        for wrapped_line in wrapped_lines:
            if y <= margin + 60:
                draw_footer()
                c.showPage()
                draw_header(first_page=False)
                c.setFont("Helvetica", 12)
                y = height - margin
                first_page = False
            c.drawString(margin, y, wrapped_line)
            y -= line_height

    # Additional Report Section
    additional_lines = [
        "The above healthcare insights were generated based on your query,",
        "aggregated from trusted sources. Please consult a healthcare professional",
        "for a full diagnosis and treatment plan."
    ]
    if y <= margin + 100:
        draw_footer()
        c.showPage()
        draw_header(first_page=False)
        y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Additional Report Details:")
    y -= 20
    c.setFont("Helvetica", 12)
    for line in additional_lines:
        wrapped_lines = textwrap.wrap(line, width=max_chars_per_line)
        for wrapped_line in wrapped_lines:
            if y <= margin + 60:
                draw_footer()
                c.showPage()
                draw_header(first_page=False)
                c.setFont("Helvetica", 12)
                y = height - margin
            c.drawString(margin, y, wrapped_line)
            y -= line_height

    draw_footer()
    c.save()
    buffer.seek(0)
    return buffer


# -------------------------------------------------------------------------------------------------------------------
# SQLite database for storing user history
DB_FILE = "context.db"


def initialize_db():
    conn = sqlite3.connect("context.db")
    c = conn.cursor()
    c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL
            )
        ''')
    c.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                feature TEXT,
                user_input TEXT,
                response TEXT,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        ''')
    conn.commit()
    conn.close()


def get_or_create_user(email):
    conn = sqlite3.connect("context.db")
    c = conn.cursor()
    c.execute("SELECT user_id FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    if result:
        user_id = result[0]
    else:
        c.execute("INSERT INTO users (email) VALUES (?)", (email,))
        conn.commit()
        user_id = c.lastrowid
    conn.close()
    return user_id


def save_user_query(user_id, feature, user_input, response):
    conn = sqlite3.connect("context.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (user_id, feature, user_input, response) VALUES (?, ?, ?, ?)",
        (user_id, feature, user_input, response)
    )
    conn.commit()
    conn.close()


def update_last_response(user_id, updated_response):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # SQLite doesn't support ORDER BY with LIMIT in UPDATE so a workaround is needed.
    # Here we update the most recent entry (max(id)) for the user.
    c.execute(
        "UPDATE history SET response = ? WHERE id = (SELECT id FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 1)",
        (updated_response, user_id)
    )
    conn.commit()
    conn.close()


def get_user_history(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT feature, user_input, response FROM history WHERE user_id = ? ORDER BY id DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows


# Helper: load history for a given feature
def load_history_for_feature(user_id, feature_name):
    rows = get_user_history(user_id)
    history = []
    for row in rows:
        if row[0] == feature_name:
            history.append({"query": row[1], "response": row[2]})
    return history


def display_history():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows


# ---------------------------
# Get Doctors from Database
# ---------------------------
def get_doctors():
    conn = sqlite3.connect('doctors.db')
    c = conn.cursor()
    c.execute('SELECT name, email FROM doctors')
    doctors = c.fetchall()
    conn.close()
    return doctors

def send_email_sendgrid(recipient_email, subject, body, attachment_buffer):
    sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)

    from_email = Email("rshbhsharma.27@gmail.com")
    to_email = To(recipient_email)
    content = Content("text/plain", body)
    mail = Mail(from_email, to_email, subject, content)

    attachment_buffer.seek(0)
    encoded_file = base64.b64encode(attachment_buffer.read()).decode()

    attached_file = Attachment()
    attached_file.file_content = FileContent(encoded_file)
    attached_file.file_type = FileType("application/pdf")
    attached_file.file_name = FileName("medical_report.pdf")
    attached_file.disposition = Disposition("attachment")

    mail.attachment = attached_file

    try:
        response = sg.send(mail)
        print(f"Email sent with status code: {response.status_code}")
        return response
    except Exception as e:
        print(f"Error sending email: {e}")
        return None


initialize_db()

# import pandas as pd

# Load severity data from CSV
severity_df = pd.read_csv('severity.csv')
# Create a dictionary for symptom severity lookup
severity_dict = dict(zip(severity_df['Symptom'], severity_df['Severity']))

# --------------------------
# --- Main App Navigation & UI ---

# Sidebar Navigation for overall app.
st.sidebar.markdown("<h2 style='color: #ffffff;'>📌 Navigation</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-text'>Use the sidebar to explore different features.</p>",
                    unsafe_allow_html=True)
st.sidebar.image("utils/main_profile.png", use_container_width=True)

# --- Hardcoded SendGrid API Key ---
SENDGRID_API_KEY = "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"
FROM_EMAIL = "pratyush.20mis7028@vitap.ac.in"  # Use verified sender email


# ---------------------------------------
# Main App
# ---------------------------------------
def main():

    # Initialize DB
    initialize_db()

    # Get user email for history purposes
    email = st.text_input("Enter your email:", placeholder="you@example.com")
    if not email:
        st.info("Please enter your email to continue.")
        st.stop()
    user_id = get_or_create_user(email)

    # Load conversation histories for each feature from the DB if not already in session_state
    if "conversation_history_disease" not in st.session_state:
        st.session_state["conversation_history_disease"] = load_history_for_feature(user_id,
                                                                                    "Disease Diagnosis & Recommendation")
    if "conversation_history_prescription" not in st.session_state:
        st.session_state["conversation_history_prescription"] = load_history_for_feature(user_id,
                                                                                         "Healthcare Prescription Insight")

    # Sidebar: Feature selection
    feature = st.sidebar.selectbox("Choose Feature",
                                   ["Disease Prediction & Recommendation", "Healthcare Prescription Insight"])

    # Display Header based on Feature
    if feature == "Disease Prediction & Recommendation":
        st.markdown("<div class='title'>Advanced Diagnosis & Treatment Recommendations</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='title'>Healthcare Prescription Insight</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Feature 1: Disease Prediction & Recommendation
    if feature == "Disease Prediction & Recommendation":
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Disease Prediction from Symptoms")
        st.markdown("_For best accuracy, select as many symptoms as possible._")
        predicted_diseases_list = []

        # Load ML models
        model_disease = load_model('models/our_new_models/disease_model.h5')
        tokenizer = joblib.load('models/our_new_models/tokenizer.pkl')
        label_encoder = joblib.load('models/our_new_models/label_encoder.pkl')

        all_symptoms = sorted(list(tokenizer.word_index.keys()))
        st.title("🩺 Disease Prediction from Symptoms")
        selected_symptoms = st.multiselect(
            "Select your symptoms:",
            all_symptoms,
            help="Select one or more symptoms from the list."
        )
        st.session_state["selected_symptoms"] = selected_symptoms

        @st.cache_data()
        def load_severity_data():
            df = pd.read_csv('severity.csv')
            return {row['Symptom'].strip().lower(): row['Severity'].strip() for _, row in df.iterrows()}

        severity_dict = load_severity_data()
        if selected_symptoms:
            st.subheader("🧭 Severity of Selected Symptoms")
            severity_color = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71", "Unknown": "#bdc3c7"}
            for symptom in selected_symptoms:
                severity = severity_dict.get(symptom.strip().lower(), "Unknown")
                st.markdown(
                    f"<div style='padding:6px 12px; border-radius:8px; background-color:{severity_color.get(severity)}; color:white; margin-bottom:6px; font-weight:bold;'>"
                    f"🔹 {symptom} — {severity} Severity</div>", unsafe_allow_html=True
                )

        if st.button("🔍 Predict Disease"):
            if selected_symptoms:
                input_text = ','.join(selected_symptoms)
                seq = tokenizer.texts_to_sequences([input_text])
                padded = pad_sequences(seq, maxlen=model_disease.input_shape[1], padding='post')
                preds = model_disease.predict(padded)
                top_indices = preds[0].argsort()[::-1]
                confident_preds = [
                    (label_encoder.inverse_transform([i])[0], preds[0][i] * 100)
                    for i in top_indices if preds[0][i] * 100 > 85
                ]
                if confident_preds:
                    st.subheader("✅ High Confidence Predictions (>85%)")
                    predicted_diseases_list = []
                    for disease, conf in confident_preds:
                        st.write(f"🩺 **{disease}** — 💡 Confidence: **{conf:.2f}%**")
                        predicted_diseases_list.append(disease)
                    st.session_state["predicted_diseases_list"] = predicted_diseases_list

                    severity_score_map = {"High": 3, "Medium": 2, "Low": 1}
                    total_score = sum(severity_score_map.get(severity_dict.get(sym.strip().lower(), "Unknown"), 0)
                                      for sym in selected_symptoms)
                    st.markdown("### 🧾 Visit Recommendation Based on Severity")
                    if total_score >= 8:
                        st.error(
                            "🚨 Based on your symptom severity, it's strongly advised to visit a hospital immediately.")
                    elif total_score >= 5:
                        st.warning("⚠️ Your symptoms indicate a need to consult a doctor soon.")
                    else:
                        st.success("✅ Your symptoms seem mild. Home care may be sufficient for now.")
                else:
                    st.warning("No disease reached 85% confidence. Please provide additional details or symptoms.")
            else:
                st.warning("Please select at least one symptom.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Disease Description and Recommendations")

        def format_symptom_severity():
            if "selected_symptoms" in st.session_state:
                return [f"{sym} ({severity_dict.get(sym.strip().lower(), 'Unknown')})"
                        for sym in st.session_state["selected_symptoms"]]
            return []

        def calculate_severity_score():
            severity_score_map = {"High": 3, "Medium": 2, "Low": 1}
            return sum(severity_score_map.get(severity_dict.get(sym.strip().lower(), "Unknown"), 0)
                       for sym in st.session_state["selected_symptoms"])

        if st.session_state.get("predicted_diseases_list"):
            predicted_diseases_str = "\n".join(st.session_state["predicted_diseases_list"])
            st.markdown("**Predicted Diseases:**")
            st.write(predicted_diseases_str)
            symptom_severity_strs = format_symptom_severity()
            combined_query = f"Symptoms: {', '.join(symptom_severity_strs)}\nPredicted Diseases: {predicted_diseases_str}"
            total_score = calculate_severity_score()
            if total_score >= 8:
                severity_warning = "🚨 Based on your symptom severity, it's strongly advised to visit a hospital immediately."
            elif total_score >= 5:
                severity_warning = "⚠️ Your symptoms indicate a need to consult a doctor soon."
            else:
                severity_warning = "✅ Your symptoms seem mild. Home care may be sufficient for now."
            combined_query += f"\nSeverity Warning: {severity_warning}"
            custom_prompt = (
                "\n\nInstruction: Act as a home diagnostic assistant for people living in remote areas with limited access to healthcare facilities. "
                "Using only the provided symptoms, their severity, and the predicted diseases, generate a structured response divided into the following sections:\n\n"
                "1. **About the Disease**: Provide a brief, non-technical explanation of each predicted disease.\n"
                "2. **Symptom Mapping**: Match the symptoms (with severity) to the predicted diseases(MANDATORY) by Hook or Crook and make it sound logical dont start with that itrs not directly related tell, come to the point directly .\n"
                "3. **Home Diagnosis & Medication**: Suggest basic home checks and over-the-counter medicines by giving Salt name (avoid brand names) that could help.\n"
                "4. **Consult a Doctor**: Recommend if/when to seek medical help, explaining the reasoning. if doctor visit is recommende then tell to go Healthcare prescription Insight tab of CoDiagnose App\n"

                "Do not include extra information. Be empathetic, practical, and medically grounded."
            )
            combined_query += custom_prompt
            # st.text_area("Full Query:", combined_query, height=150)
            response = process_query(combined_query, predicted_diseases=predicted_diseases_str)
            st.session_state["disease_recommendation"] = response  # Persist the RAG answer dedicated to Feature 1
            st.subheader("Disease Recommendations:")
            st.write(response)
            # save_user_query(user_id, "Disease Diagnosis & Recommendation", combined_query, response)

        else:
            st.warning("Please select symptoms or provide additional details.")
            additional_details = st.text_input("Enter additional details (e.g., persistent fever, severe headache):",
                                               placeholder="Type additional context here...")
            if st.button("Continue for Detailed Recommendation"):
                if additional_details:
                    symptom_severity_strs = format_symptom_severity()
                    base_info = f"Symptoms: {', '.join(symptom_severity_strs)}"
                    combined_query = f"{base_info}\nAdditional Info: {additional_details}"
                    total_score = calculate_severity_score()
                    if total_score >= 8:
                        severity_warning = "🚨 Based on your symptom severity, it's strongly advised to visit a hospital immediately."
                    elif total_score >= 5:
                        severity_warning = "⚠️ Your symptoms indicate a need to consult a doctor soon."
                    else:
                        severity_warning = "✅ Your symptoms seem mild. Home care may be sufficient for now."
                    combined_query += f"\nSeverity Warning: {severity_warning}"
                    custom_prompt = (
                        "\n\nYou are a home diagnostic assistant for people living in remote areas with limited access to healthcare facilities. "
                        "Your goal is to provide early guidance based on the user's symptoms.\n\n"
                        "The user provides two types of symptoms:\n"
                        "- Regular symptoms with severity tags (High, Medium, Low).\n"
                        "- Additional Info: symptoms without severity tags, which may offer critical hints and should be given very high importance.\n\n"

                        "Based strictly on the provided symptoms and their severity, predict the top two most likely diseases. "
                        "Your prediction must prioritize the symptoms with severity tags, but it is MANDATORY to logically map the additional info symptoms to the diseases — by hook or crook. "
                        "Do not ignore or dismiss any symptom.\n\n"

                        "Structure the response into the following clear sections:\n\n"
                        "1. 🩺 About the Predicted Diseases:\n"
                        "   - Provide a short, easy-to-understand explanation of each predicted disease.\n\n"
                        "2. 🔍 Symptom Mapping:\n"
                        "   - Match each of the given symptoms (both severity-tagged and additional info) to the predicted diseases.\n"
                        "   - Be direct and logical. Do not say 'this symptom is not directly related'. Instead, explain how it could be relevant.\n\n"
                        "3. 🏠 Home Diagnosis & Medication:\n"
                        "   - Suggest any basic checks or self-assessments that can be done at home.\n"
                        "   - Recommend over-the-counter medication using only generic salt names (not brand names), preferably common in India.\n\n"
                        "4. 👨‍⚕️ Consult a Doctor:\n"
                        "   - Clearly state whether a doctor's visit is necessary or not.\n"
                        "   - if yes then tell to go Healthcare prescription Insight tab of CoDiagnose App\n"
                        "   - If yes, explain why, and suggest what can be done temporarily to manage the situation until medical help is reached.\n\n"

                        "Be concise, medically grounded, practical, and empathetic. Do not include anything outside the given structure."
                    )

                    combined_query += custom_prompt
                    st.text_area("Full Query:", combined_query, height=150)
                    response = process_query(combined_query)
                    st.session_state["disease_recommendation"] = response  # Persist Feature 1 answer here
                    st.subheader("Disease Recommendations:")
                    st.write(response)
                    # save_user_query(user_id, "Disease Diagnosis & Recommendation", combined_query, response)
                else:
                    st.warning("Please provide additional details before continuing.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section">', unsafe_allow_html=True)
        # Extra Section: Symptom and Medicine Lookup (CSV Based) – Appears only if RAG response exists
        if st.session_state.get("disease_recommendation"):
            st.markdown("---")
            st.markdown("### Symptom and Medicine Lookup")
            df = pd.read_csv("data_for_overall_sys/Symptoms_with_meds_updated.csv")
            df["Suggested Medicine"] = df["Suggested Medicine"].replace("Not Available",
                                                                        "Not available as symptom severity is high")
            df["1mg Link"] = df["1mg Link"].replace("Not Available", "Not available as symptom severity is high")
            df["Description"] = df["Description"].replace("Not Available", "Not available as symptom severity is high")

            st.write("Below are the suggested medicines and details for the symptoms you selected earlier:")
            # Use stripped versions of the symptom names from the CSV
            symptom_options = [sym.strip() for sym in df['Symptom'].unique().tolist()]
            # Get the default symptoms from session_state (strip them too)
            default_symptoms = [sym.strip() for sym in st.session_state.get("selected_symptoms", [])]

            selected_symptoms_csv = st.multiselect(
                "Select Symptoms (CSV Lookup)",
                options=symptom_options,
                default=default_symptoms
            )

            if selected_symptoms_csv:
                # Ensure you match lower-case, stripped values
                df["Symptom_lower"] = df["Symptom"].str.strip().str.lower()
                selected_symptoms_lower = [sym.lower() for sym in selected_symptoms_csv]
                filtered_df = df[df["Symptom_lower"].isin(selected_symptoms_lower)]

                high_severity_df = filtered_df[filtered_df["Severity"].str.lower() == "high"]
                low_medium_severity_df = filtered_df[filtered_df["Severity"].str.lower().isin(["low", "medium"])]

                if not low_medium_severity_df.empty:
                    low_medium_severity_df = low_medium_severity_df.drop(columns=["Severity", "Symptom_lower"])
                    low_medium_severity_df["1mg Link"] = low_medium_severity_df["1mg Link"].apply(
                        lambda x: f'<a href="{x}" target="_blank">Link</a>' if "not available" not in x.lower() else x)
                    st.write("### Suggested Medicine and Details for Low/Medium Severity Symptoms")
                    st.markdown(low_medium_severity_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                if not high_severity_df.empty:
                    st.write("**Medicines are Not Available for High Severity Symptoms**")
            else:
                st.write("No symptoms were selected above. Please select symptoms to get suggestions.")

        # PART D: Additional Doctor/Patient Chat
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Doctor/Patient Chat")
        additional_query = st.text_input("Ask any further questions you have for the doctor:",
                                         placeholder="Type your query here...")
        if st.button("Send Query"):
            from langchain.chat_models import ChatOpenAI
            openai_api_key = st.secrets.get("OPENAI_API_KEY")
            doctor_model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o", temperature=0)
            doctor_response = doctor_model.predict(additional_query)
            st.markdown("**Doctor's Response:**")
            st.write(doctor_response)
        st.markdown("""
            <hr>
            <p style='text-align: center; color: #7f8c8d; font-family: Roboto, sans-serif;'>
                Made by <span style='color: #007bff;'>Pratyush Puri Goswami</span> | <span style='color: #6f42c1;'>Rishabh Sharma</span>
            </p>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # Feature 2: Healthcare Prescription Insight
    elif feature == "Healthcare Prescription Insight":
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Medical Report Generator")
        st.markdown("Fill in the details below to generate a downloadable PDF medical report.")
        name = st.text_input("Name", key="presc_name")
        age = st.number_input("Age", min_value=0, max_value=150, step=1, key="presc_age")
        gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"], key="presc_gender")
        st.markdown('</div>', unsafe_allow_html=True)

        primary_concern = st.text_input("🩺 What is the primary medical concern?", key="primary_concern")

        if primary_concern:
            st.markdown("#### Additional Medical Context")

            # Radio 1: Current Medications
            med_history = st.radio("Have you already mentioned your current medications?", ["Yes", "No"],
                                   key="radio_med")
            current_meds_text = ""
            if med_history == "No":
                current_meds_text = st.text_input("Please enter your current medications:", key="current_meds_text")

            # Radio 2: Family History
            family_history = st.radio("Have you already included any family medical history?", ["Yes", "No"],
                                      key="radio_fam")
            family_hist_text = ""
            if family_history == "No":
                family_hist_text = st.text_input("Please enter any relevant family history:", key="family_hist_text")

            # Radio 3: Additional Info
            additional_info = st.radio("Have you included all the Symptoms clearly with their Durations", ["Yes", "No"],
                                       key="radio_additional")
            additional_info_text = ""
            if additional_info == "No":
                additional_info_text = st.text_input("Please enter any additional relevant info:",
                                                     key="additional_info_text")

            # Biomedical Query Text Box
            user_query_raw = st.text_area("📝Add Some more Details which will be helpful for the Doctor",
                                          key="presc_query")

            # Combine all the info into a single query
            combined_query = f"Primary Concern: {primary_concern}\n"
            if user_query_raw:
                combined_query += f"{user_query_raw}\n"
            if current_meds_text:
                combined_query += f"Current Medications: {current_meds_text}\n"
            if family_hist_text:
                combined_query += f"Family History: {family_hist_text}\n"
            if additional_info_text:
                combined_query += f"Additional Info: {additional_info_text}\n"

        if st.button("Get RAG based prescription and insight", key="presc_btn") and combined_query:
            last_context = ""
            if st.session_state.get("conversation_history_prescription"):
                last_turn = st.session_state["conversation_history_prescription"][-1]
                last_context = last_turn["response"]

            custom_prompt = textwrap.dedent(f"""
    You are an experienced clinical consultant with over 20 years of expertise in diagnostics and patient management. 
    Your task is to generate a medically accurate, structured diagnostic report that synthesizes the patient's clinical presentation, history, and medical context, aiding the healthcare professional in decision-making.

    The response should adhere to the following structure:

    1. **Clinical Summary:**  
       – Provide a succinct summary of the patient's clinical presentation, incorporating the presenting symptoms and any relevant medical history.  
       – Utilize appropriate medical terminology to describe the pathophysiology, clinical manifestations, and associated risks of the condition.

    2. **Differential Diagnosis (Two Conditions):**  
       – Propose two likely differential diagnoses based on the clinical data.  
       – Clearly distinguish between these diagnoses using specific clinical features (e.g., pathognomonic signs, laboratory results, progression of disease).

    3. **Symptom-Disease Mapping:**  
       – For each proposed differential diagnosis, systematically map the reported symptoms (both severity-tagged and non-severity-tagged) to the potential pathophysiological mechanisms underlying each disease.  
       – Consider the temporal progression, severity, and nature of symptoms in relation to the pathogenesis of each condition.

    4. **Diagnostic Workup:**  
       – Recommend a comprehensive diagnostic workup to confirm the working diagnosis.  
       – Include laboratory investigations (e.g., Complete Blood Count, liver function tests, renal function tests, inflammatory markers, specific antibodies, PCR), imaging modalities (e.g., Chest X-ray, CT scan, MRI), and any disease-specific diagnostic tests (e.g., ECG, biopsy, culture).  
       – Provide timing for each test and the clinical rationale for selecting these investigations.

    5. **Therapeutic Plan:**  
       – Based on the likely diagnosis, present a detailed therapeutic regimen, including:  
           – **Pharmacological Treatments:**  
               – Medication names (both generic and brand, where applicable).  
               – Active pharmaceutical ingredients, dosages, routes of administration, and treatment durations.  
               – Highlight known contraindications, potential adverse effects, and any required monitoring parameters (e.g., renal/hepatic function, drug interactions).  
           – **Non-pharmacological Interventions:**  
               – Lifestyle modifications (e.g., dietary recommendations, physical activity, stress management).  
               – Complementary treatments (e.g., physical therapy, occupational therapy, psychological counseling).  
               – Supportive care (e.g., oxygen therapy, intravenous hydration, wound care).

    6. **Disease Staging & Prognosis:**  
       – If applicable, define the stages of the disease (e.g., staging for cancer, heart failure, or chronic kidney disease) and evaluate the severity based on clinical and laboratory markers.  
       – Provide a prognosis based on the stage, emphasizing factors that may influence disease progression, response to treatment, or long-term outcomes.

    7. **Follow-Up and Monitoring Plan:**  
       – Outline a follow-up plan, detailing the frequency of subsequent evaluations, the need for re-assessment of clinical parameters, and indications for further investigations or specialist referrals.  
       – Include any red flags or critical symptoms that require immediate intervention (e.g., organ failure, sepsis, acute exacerbations).

    8. **Referral Recommendations:**  
       – Indicate the need for referral to relevant specialists (e.g., cardiology, oncology, neurology) based on the differential diagnosis.  
       – Provide criteria for referral, particularly for cases requiring advanced diagnostics, management, or second opinions.

    9. **Patient Education and Risk Management:**  
       – Provide relevant patient education on the disease process, treatment options, and lifestyle modifications.  
       – Discuss the potential risks of non-compliance with treatment, the importance of follow-up, and any necessary adjustments to patient behavior or environment to improve outcomes.

    **Instructions:**  
    – Strictly base your response on the patient’s clinical data, history, and symptoms as provided.  
    – Ensure that all medical details are scientifically accurate and consistent with the latest clinical guidelines and best practices.  
    – Refrain from including any extraneous information or personal assumptions.  
      Maintain a tone suitable for professional medical practitioners, avoiding layman terminology.

    --- Patient Information ---  
    Name: {name or "N/A"}  
    Age: {age or "N/A"}  
    Gender: {gender if gender != "Select" else "N/A"}  

    --- Current Status Query ---  
    {combined_query}

    ---  Medical Status Context ---  
    {last_context or "None"}  



    --- END OF INSTRUCTIONS ---
""")

            # st.text_area("Combined Query Prompt", custom_prompt, height=300)
            answer = process_query(custom_prompt, custom_template=custom_prompt)
            if answer:
                st.session_state["prescription_insight"] = answer
                if "conversation_history_prescription" not in st.session_state:
                    st.session_state["conversation_history_prescription"] = []
                st.session_state["conversation_history_prescription"].append(
                    {"query": combined_query, "response": answer})
                if len(st.session_state["conversation_history_prescription"]) > 5:
                    st.session_state["conversation_history_prescription"] = st.session_state[
                                                                                "conversation_history_prescription"][
                                                                            -5:]
                save_user_query(user_id, "Healthcare Prescription Insight", combined_query, answer)
        if st.session_state.get("prescription_insight"):
            st.write("**Response:**")
            st.write(st.session_state["prescription_insight"])


        st.markdown("---")
        st.markdown("### Generate and Send Report")
        # Get doctor information from the doctors.db
        doctors = get_doctors()
        if doctors:
            doctor_names = [doc[0] for doc in doctors]
            selected_doctor = st.selectbox("Select Doctor", doctor_names, key="presc_doctor")
            recipient_email = next((email for name, email in doctors if name == selected_doctor), None)
            if st.button("Generate and Send Report", key="presc_send_btn"):
                if name and age and gender != "Select" and recipient_email:
                    insight = st.session_state.get("prescription_insight", "No insights available.")
                    pdf_buffer = generate_medical_report(name, age, gender, insight)
                    pdf_bytes = pdf_buffer.getvalue()
                    st.download_button("📥 Download PDF", data=pdf_bytes, file_name=f"{name}_medical_report.pdf",
                                       mime="application/pdf")
                    send_email_sendgrid(
                        recipient_email,
                        subject=f"Medical Report for {name}",
                        body=f"Dear Doctor,\n\nPlease find the attached medical report for patient {name}.\n\nRegards,\nCoDiagnose Team",
                        attachment_buffer=io.BytesIO(pdf_bytes)
                    )
                    st.success("✅ Report generated and emailed successfully.")
                else:
                    st.error("Please fill in all fields and select a doctor.")
        else:
            st.warning("No doctors found in database. Please insert doctors manually in 'doctors.db' SQLite file.")


if __name__ == "__main__":
    main()