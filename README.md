
# AI Resume Screening and Ranking

A Streamlit-based web application that ranks resumes based on their similarity to a provided job description using TF-IDF and cosine similarity. The app extracts text from uploaded PDF resumes, calculates similarity scores, and provides visualizations and export options.

## Features
- Upload multiple PDF resumes and enter a job description.
- Ranks resumes by similarity to the job description (cosine similarity).
- Displays top N candidates with similarity scores.
- Visualizes score distribution with a bar plot.
- Extracts key skills using SpaCy (optional).
- Exports results in CSV, Excel, or JSON format.
- Highlights matching keywords in resume text.

## Demo
Deployed on Streamlit Community Cloud: [AI Resume Ranker](https://ai-resume-screening-and-ranking-qufydjrajcos52qebsdben.streamlit.app/)

## Prerequisites
- Python 3.8+
- Git
- Streamlit Cloud account (for deployment)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/ai-resume-screening-and-ranking.git
   cd ai-resume-screening-and-ranking
   ```

2. **Set Up a Virtual Environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Requirements
Create a `requirements.txt` file with the following:
```
streamlit==1.43.2
PyPDF2==3.0.1
pandas==2.2.3
scikit-learn==1.5.2
spacy==3.8.2
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
matplotlib==3.9.2
seaborn==0.13.2
```

## Usage

1. **Run the App Locally**
   ```bash
   streamlit run resumeranking_app.py
   ```
   Open your browser at `http://localhost:8501`.

2. **How to Use**
   - Enter a job description in the text area.
   - Upload one or more PDF resumes via the file uploader.
   - Adjust options in the sidebar:
     - Minimum similarity score (0.0–1.0).
     - Number of top candidates to display (1–50).
     - Enable/disable skill extraction.
     - Choose export format (CSV, Excel, JSON).
   - View ranked results, a bar plot, detailed matches, and export the results.

## Deployment on Streamlit Cloud
1. Push the repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Create a new app:
   - Select your repository (`ai-resume-screening-and-ranking`).
   - Set the branch to `main`.
   - Specify the main file as `resumeranking_app.py`.
4. Deploy the app. It will install dependencies from `requirements.txt` automatically.

## Troubleshooting
- **PDF Errors**: Ensure uploaded PDFs contain extractable text (not scanned images).
- **SpaCy Issues**: If skill extraction fails, verify the SpaCy model URL in `requirements.txt` or disable the feature.
- **Plot Errors**: Ensure `matplotlib` and `seaborn` are installed and resumes have valid text.

## Limitations
- Skill extraction requires SpaCy and the `en_core_web_sm` model.
- Only supports PDF resumes with extractable text.
- Performance may vary with large numbers of resumes due to TF-IDF computation.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- Built with [Streamlit](https://streamlit.io/), [PyPDF2](https://github.com/py-pdf/PyPDF2), and [scikit-learn](https://scikit-learn.org/).
- Icons by [Flaticon](https://www.flaticon.com/).
```

---

### Instructions
1. Create the File:
   - Save the above content as `README.md` in the root of your repository (`ai-resume-screening-and-ranking`).

2. Customize:
   - Replace `your-username` in the clone URL with your actual GitHub username.
   - Update the demo link if your Streamlit Cloud URL changes.
   - Add a `LICENSE` file if you want to include one (e.g., MIT License text).

3. Push to GitHub:
   ```bash
   git add README.md
   git commit -m "Add README file"
   git push origin main
   ```

