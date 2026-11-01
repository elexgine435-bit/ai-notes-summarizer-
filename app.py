"""AI Notes Summarizer: Streamlit app."""

import streamlit as st

from utils import ai_engine as ai
from utils.pdf_reader import PDFError, extract_text_from_pdf

# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Notes Summarizer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    padding: 2rem 2.2rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
}
.hero-title { color: #ffffff; font-size: 2.2rem; font-weight: 800; line-height: 1.2; }
.hero-sub   { color: #E0E7FF; font-size: 1.05rem; margin-top: 0.5rem; }
.page-title { color: #3730A3; font-size: 1.9rem; font-weight: 800; margin-bottom: 0.1rem; }
.page-sub   { color: #64748B; font-size: 1rem; margin-bottom: 1.2rem; }
.sidebar-brand {
    font-size: 1.4rem; font-weight: 800; color: #3730A3;
    line-height: 1.25; padding: 0.4rem 0 0.8rem 0;
}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 10px; font-weight: 600; padding: 0.55rem 1.3rem;
}
[data-testid="stMetricValue"] { color: #4F46E5; }
@media (max-width: 640px) {
    .hero { padding: 1.2rem; }
    .hero-title { font-size: 1.5rem; }
    .page-title { font-size: 1.5rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state (the app's memory between clicks)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "doc_text": "",
    "doc_name": "",
    "doc_pages": 0,
    "doc_id": None,
    "summary": "",
    "key_points": "",
    "mcqs": [],
    "quiz_submitted": False,
    "flashcards": [],
    "card_index": 0,
    "show_answer": False,
    "chat": [],
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def clear_quiz_widgets():
    for key in list(st.session_state.keys()):
        if key.startswith("mcq_"):
            del st.session_state[key]


def load_new_document(file_id, name, text, pages):
    """Store a new document and wipe results from the previous one."""
    st.session_state.doc_id = file_id
    st.session_state.doc_name = name
    st.session_state.doc_text = text
    st.session_state.doc_pages = pages
    st.session_state.summary = ""
    st.session_state.key_points = ""
    st.session_state.mcqs = []
    st.session_state.quiz_submitted = False
    st.session_state.flashcards = []
    st.session_state.card_index = 0
    st.session_state.show_answer = False
    st.session_state.chat = []
    clear_quiz_widgets()


def page_header(title, subtitle):
    st.markdown(
        f"<div class='page-title'>{title}</div><div class='page-sub'>{subtitle}</div>",
        unsafe_allow_html=True,
    )


def require_document():
    if not st.session_state.doc_text:
        st.warning("📂 Please upload a PDF first on the **Home & Upload** page.")
        st.stop()


# ---------------------------------------------------------------------------
# Page: Home & Upload
# ---------------------------------------------------------------------------
def page_home():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🎓 AI Notes Summarizer</div>
            <div class="hero-sub">Upload your PDF notes and study smarter with summaries,
            key points, quizzes, flashcards and an AI tutor.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📤 Upload your notes")
    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Use a PDF with selectable text (not a scanned photo).",
    )

    if uploaded is not None:
        file_id = f"{uploaded.name}-{uploaded.size}"
        if file_id != st.session_state.doc_id:
            with st.spinner("Reading your PDF..."):
                try:
                    text, pages = extract_text_from_pdf(uploaded)
                except PDFError as error:
                    st.error(f"❌ {error}")
                    return
            load_new_document(file_id, uploaded.name, text, pages)
            st.success("✅ PDF uploaded successfully! Choose a tool from the sidebar.")

    if st.session_state.doc_text:
        text = st.session_state.doc_text
        st.markdown(f"#### 📄 {st.session_state.doc_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pages", st.session_state.doc_pages)
        c2.metric("Words", f"{len(text.split()):,}")
        c3.metric("Characters", f"{len(text):,}")

        if len(text) > ai.MAX_CHARS:
            st.info(
                f"ℹ️ Your document is long. Summaries, key points, MCQs and flashcards use the "
                f"first {ai.MAX_CHARS:,} characters. The Ask Questions tool searches the whole document."
            )
        with st.expander("👀 Preview extracted text"):
            st.text(text[:2000] + ("..." if len(text) > 2000 else ""))
    else:
        st.markdown("### ✨ What you can do")
        cols = st.columns(3)
        features = [
            ("📝 Smart Summary", "Short, organized summaries with headings."),
            ("⭐ Important Points", "Definitions, concepts and key facts."),
            ("🧠 MCQ Quiz", "Practice questions with instant scoring."),
            ("💬 Ask Questions", "Chat with your notes like a tutor."),
            ("🃏 Flashcards", "Flip cards for fast revision."),
            ("🔒 Private", "Your notes are processed only while you use the app."),
        ]
        for i, (title, desc) in enumerate(features):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(desc)


# ---------------------------------------------------------------------------
# Page: Summary
# ---------------------------------------------------------------------------
def page_summary():
    page_header("📝 Smart Summary", "Get a short, easy-to-read summary organized by topic.")
    require_document()

    detail = st.select_slider(
        "Summary length", options=["Brief", "Standard", "Detailed"], value="Standard"
    )
    if st.button("✨ Generate Summary", type="primary"):
        with st.spinner("Writing your summary..."):
            try:
                st.session_state.summary = ai.generate_summary(st.session_state.doc_text, detail)
            except ai.AIError as error:
                st.error(f"❌ {error}")

    if st.session_state.summary:
        with st.container(border=True):
            st.markdown(st.session_state.summary)
        st.download_button(
            "⬇️ Download summary",
            st.session_state.summary,
            file_name="summary.md",
            mime="text/markdown",
        )


# ---------------------------------------------------------------------------
# Page: Important points
# ---------------------------------------------------------------------------
def page_key_points():
    page_header("⭐ Important Points", "Key definitions, concepts and facts from your notes.")
    require_document()

    if st.button("🔍 Extract Important Points", type="primary"):
        with st.spinner("Finding the most important points..."):
            try:
                st.session_state.key_points = ai.generate_key_points(st.session_state.doc_text)
            except ai.AIError as error:
                st.error(f"❌ {error}")

    if st.session_state.key_points:
        with st.container(border=True):
            st.markdown(st.session_state.key_points)
        st.download_button(
            "⬇️ Download key points",
            st.session_state.key_points,
            file_name="key_points.md",
            mime="text/markdown",
        )


# ---------------------------------------------------------------------------
# Page: MCQs
# ---------------------------------------------------------------------------
def option_labels(question):
    return [f"{'ABCD'[i]}. {opt}" for i, opt in enumerate(question["options"])]


def show_quiz_results(mcqs):
    results = []
    for i, q in enumerate(mcqs):
        labels = option_labels(q)
        correct = labels[q["answer_index"]]
        chosen = st.session_state.get(f"mcq_{i}")
        results.append((q, chosen, correct))

    score = sum(1 for _, chosen, correct in results if chosen == correct)
    st.markdown("### 🏆 Your Results")
    st.metric("Score", f"{score} / {len(mcqs)}")
    if score == len(mcqs):
        st.success("Perfect score! Excellent work! 🎉")
    elif score >= len(mcqs) / 2:
        st.info("Good job! Review the explanations below to improve further.")
    else:
        st.warning("Keep practicing! Read the explanations below and try again.")

    for i, (q, chosen, correct) in enumerate(results, start=1):
        icon = "⚪" if chosen is None else ("✅" if chosen == correct else "❌")
        with st.expander(f"{icon} Question {i}: {q['question']}"):
            st.markdown(f"**Your answer:** {chosen if chosen else 'Not answered'}")
            st.markdown(f"**Correct answer:** {correct}")
            if q["explanation"]:
                st.markdown(f"**Why:** {q['explanation']}")


def page_mcq():
    page_header("🧠 MCQ Practice", "Test yourself with multiple-choice questions from your notes.")
    require_document()

    count = st.slider("Number of questions", min_value=3, max_value=15, value=5)
    if st.button("🎯 Generate MCQs", type="primary"):
        with st.spinner("Creating your quiz..."):
            try:
                st.session_state.mcqs = ai.generate_mcqs(st.session_state.doc_text, count)
                st.session_state.quiz_submitted = False
                clear_quiz_widgets()
            except ai.AIError as error:
                st.error(f"❌ {error}")

    mcqs = st.session_state.mcqs
    if not mcqs:
        return

    st.markdown(f"### 📋 Quiz ({len(mcqs)} questions)")
    with st.form("quiz_form"):
        for i, q in enumerate(mcqs):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            st.radio(
                f"Question {i + 1}",
                option_labels(q),
                index=None,
                key=f"mcq_{i}",
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button("✅ Submit Quiz", type="primary")

    if submitted:
        st.session_state.quiz_submitted = True
    if st.session_state.quiz_submitted:
        show_quiz_results(mcqs)


# ---------------------------------------------------------------------------
# Page: Ask questions
# ---------------------------------------------------------------------------
def page_ask():
    page_header("💬 Ask Your Notes", "Ask any question. The AI answers using your uploaded document.")
    require_document()

    if st.session_state.chat:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat = []
            st.rerun()
    else:
        st.info("💡 Try: *What is the main idea of these notes?* or *Explain the first topic simply.*")

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your notes...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ai.answer_question(
                        st.session_state.doc_text, question, st.session_state.chat
                    )
                except ai.AIError as error:
                    answer = None
                    st.error(f"❌ {error}")
            if answer:
                st.markdown(answer)
                st.session_state.chat.append({"role": "user", "content": question})
                st.session_state.chat.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# Page: Flashcards
# ---------------------------------------------------------------------------
def go_to_card(step):
    total = len(st.session_state.flashcards)
    st.session_state.card_index = (st.session_state.card_index + step) % total
    st.session_state.show_answer = False


def flip_card():
    st.session_state.show_answer = not st.session_state.show_answer


def page_flashcards():
    page_header("🃏 Flashcards", "Flip through question-and-answer cards to revise quickly.")
    require_document()

    count = st.slider("Number of flashcards", min_value=5, max_value=20, value=10)
    if st.button("🃏 Generate Flashcards", type="primary"):
        with st.spinner("Making your flashcards..."):
            try:
                st.session_state.flashcards = ai.generate_flashcards(
                    st.session_state.doc_text, count
                )
                st.session_state.card_index = 0
                st.session_state.show_answer = False
            except ai.AIError as error:
                st.error(f"❌ {error}")

    cards = st.session_state.flashcards
    if not cards:
        return

    index = min(st.session_state.card_index, len(cards) - 1)
    card = cards[index]
    st.progress((index + 1) / len(cards), text=f"Card {index + 1} of {len(cards)}")

    with st.container(border=True):
        st.markdown("#### ❓ Question")
        st.markdown(card["question"])
        if st.session_state.show_answer:
            st.divider()
            st.markdown("#### ✅ Answer")
            st.markdown(card["answer"])

    c1, c2, c3 = st.columns(3)
    c1.button("⬅️ Previous", on_click=go_to_card, args=(-1,))
    c2.button(
        "🙈 Hide answer" if st.session_state.show_answer else "👀 Show answer",
        on_click=flip_card,
        type="primary",
    )
    c3.button("Next ➡️", on_click=go_to_card, args=(1,))

    with st.expander("📚 View all flashcards"):
        for i, c in enumerate(cards, start=1):
            st.markdown(f"**{i}. {c['question']}**")
            st.markdown(f"→ {c['answer']}")
            st.divider()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Home & Upload": page_home,
    "📝 Summary": page_summary,
    "⭐ Important Points": page_key_points,
    "🧠 MCQ Practice": page_mcq,
    "💬 Ask Questions": page_ask,
    "🃏 Flashcards": page_flashcards,
}

with st.sidebar:
    st.markdown("<div class='sidebar-brand'>🎓 AI Notes<br>Summarizer</div>", unsafe_allow_html=True)
    choice = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    if st.session_state.doc_text:
        st.success(f"📄 {st.session_state.doc_name}")
        st.caption(
            f"{st.session_state.doc_pages} pages · {len(st.session_state.doc_text.split()):,} words"
        )
    else:
        st.info("No document loaded yet.")

PAGES[choice]()
