

import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ModuleNotFoundError, OSError):
    SPACY_AVAILABLE = False
    st.warning("SpaCy not found or model not installed. Skill extraction will be disabled. Install with: 'pip install spacy' and 'python -m spacy download en_core_web_sm'")
import re
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Function to Extract Text from PDF

def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + " "
        return text.strip()
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}. Please check the file format and try again.")
        return ""

# Function to Extract Skills

def extract_skills(text):
    if SPACY_AVAILABLE:
        doc = nlp(text)
        skills = set()
        for chunk in doc.noun_chunks:
            if any(keyword in chunk.text.lower() for keyword in ["python", "java", "sql", "machine learning", "data analysis", "communication", "teamwork"]):
                skills.add(chunk.text)
        return list(skills) if skills else ["No key skills detected"]
    return ["Skill extraction unavailable"]

# Function to Rank Resumes

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(documents)
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(job_description_vector, resume_vectors).flatten()
    ranked_indices = cosine_similarities.argsort()[::-1]
    ranked_scores = cosine_similarities[ranked_indices]
    return ranked_scores, ranked_indices, vectorizer.get_feature_names_out()

# Function to Highlight Keywords

def highlight_keywords(text, keywords):
    for keyword in keywords:
        text = re.sub(f"\\b({keyword})\\b", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    return text

# Function to Create Downloadable CSV
def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="resume_ranking.csv">Download CSV</a>'

# Streamlit App Configuration
st.set_page_config(page_title="AI Resume Ranker", layout="wide", page_icon="📄")

st.markdown("""
    <style>
    .big-font {font-size: 40px !important; color: #1f77b4;}
    .subheader {color: #ff7f0e;}
    .highlight {background-color: #ffeb3b; padding: 2px 5px; border-radius: 3px;}
    .sidebar .sidebar-content {background-color: #f0f2f6;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">AI-Powered Resume Screening & Ranking</p>', unsafe_allow_html=True)

# Sidebar Customization Options
with st.sidebar:
    st.subheader("Customization Options")
    min_score = st.slider("Minimum Similarity Score", 0.0, 1.0, 0.3)
    top_n = st.number_input("Show Top N Candidates", min_value=1, max_value=50, value=5)
    show_skills = st.checkbox("Extract and Show Skills", value=SPACY_AVAILABLE)
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])

# UI Layout and Input Collection
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="subheader">Job Description</p>', unsafe_allow_html=True)
    job_description = st.text_area("Enter the job description here", height=200)
    
    st.markdown('<p class="subheader">Upload Resumes</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Drop your PDF resumes here", type=["pdf"], accept_multiple_files=True)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png", width=150)
    st.info("Upload resumes and enter a job description to rank candidates instantly!")

# Process and Display Results
if uploaded_files and job_description:
    with st.spinner("Analyzing resumes..."):
        resumes = []
        file_names = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if text:
                resumes.append(text)
                file_names.append(file.name)

        if resumes:
            scores, ranked_indices, feature_names = rank_resumes(job_description, resumes)
            results = pd.DataFrame({
                "Rank": range(1, len(scores) + 1),
                "Resume": [file_names[i] for i in ranked_indices],
                "Similarity Score": scores
            })
            results = results[results["Similarity Score"] >= min_score]

            if show_skills and SPACY_AVAILABLE:
                skills_list = [extract_skills(resumes[i]) for i in ranked_indices]
                results["Key Skills"] = [", ".join(skills) for skills in skills_list[:len(results)]]

            st.markdown('<p class="subheader">Ranked Candidates</p>', unsafe_allow_html=True)
            top_results = results.head(top_n)
            st.dataframe(top_results.style.format({"Similarity Score": "{:.2%}"}))

            st.markdown('<p class="subheader">Similarity Score Distribution</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, len(top_results) * 0.5))
            sns.barplot(x="Similarity Score", y="Resume", data=top_results, palette="Blues_d", ax=ax)
            ax.set_xlabel("Similarity Score (%)")
            ax.set_xlim(0, 1)
            for i, v in enumerate(top_results["Similarity Score"]):
                ax.text(v + 0.01, i, f"{v:.2%}", va="center")
            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("View Detailed Resume Matches"):
                for idx, row in top_results.iterrows():
                    resume_text = resumes[ranked_indices[idx]]
                    highlighted_text = highlight_keywords(resume_text[:500], job_description.split()[:10])
                    st.markdown(f"**Rank {row['Rank']}: {row['Resume']} (Score: {row['Similarity Score']:.2%})**", unsafe_allow_html=True)
                    st.markdown(f'<div>{highlighted_text}...</div>', unsafe_allow_html=True)

            st.markdown('<p class="subheader">Export Results</p>', unsafe_allow_html=True)
            if export_format == "CSV":
                st.markdown(get_csv_download_link(results), unsafe_allow_html=True)
            elif export_format == "Excel":
                buffer = BytesIO()
                results.to_excel(buffer, index=False)
                st.download_button("Download Excel", buffer.getvalue(), "resume_ranking.xlsx")
            elif export_format == "JSON":
                json_str = results.to_json(orient="records")
                st.download_button("Download JSON", json_str, "resume_ranking.json")
        else:
            st.error("No valid text extracted from uploaded PDFs.")
else:
    st.warning("Please upload resumes and enter a job description.")

