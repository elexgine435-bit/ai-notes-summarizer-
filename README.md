# 🎓 AI Notes Summarizer

An AI-powered study assistant built with Python and Streamlit. Upload PDF notes and get:

- 📝 Organized summaries (headings and subheadings)
- ⭐ Key definitions, concepts and important points
- 🧠 Multiple-choice quizzes with scoring and explanations
- 💬 A chat that answers questions using your notes
- 🃏 Flashcards for revision

## Tech Stack
Python · Streamlit · Groq API · PyPDF

## Run Locally

1. Open the project folder in a terminal.
2. Create and activate a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`
4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your
   Groq API key (free at https://console.groq.com).
5. Start the app: `streamlit run app.py`

## Deploy
Deploy for free on Streamlit Community Cloud. Add `GROQ_API_KEY` in the app's
**Secrets** settings. Never commit your real API key.

## Notes
- Works with PDFs that contain selectable text (not scanned images).
- Very long documents are shortened for AI features to stay within API limits.
- The AI model can be changed by adding `GROQ_MODEL = "model-name"` to your secrets.
