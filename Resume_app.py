import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:  # Avoid NoneType errors
            text += extracted_text + " "
    return text.strip()

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    
    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(documents)

    # Compute cosine similarity
    job_description_vector = vectors[0]  # First element is job description
    resume_vectors = vectors[1:]  # Remaining are resumes
    cosine_similarities = cosine_similarity(job_description_vector, resume_vectors).flatten()

    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    
    resumes = []  # Ensure the variable is initialized
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    if resumes:  # Ensure there are resumes before ranking
        # Rank resumes
        scores = rank_resumes(job_description, resumes)

        # Display scores
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        st.write(results)
    else:
        st.warning("No valid resumes extracted from PDFs.")
