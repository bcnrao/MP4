
"""
Airline Customer Support – Streamlit Frontend
=============================================
Requires FastAPI backend running on port 8000.

  pip install streamlit requests
  streamlit run app.py --server.port 8501
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(
    page_title="✈️ Airline Customer Support",
    page_icon="✈️",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .title   { font-size:2rem; font-weight:700; color:#1e3a8a; text-align:center; margin-bottom:0.2rem; }
  .sub     { text-align:center; color:#6b7280; margin-bottom:1.5rem; }
  .chip    { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem;
             font-weight:600; margin-right:6px; }
  .sql     { background:#dbeafe; color:#1d4ed8; }
  .rag     { background:#dcfce7; color:#166534; }
  .ooc     { background:#fef9c3; color:#854d0e; }
  .blocked { background:#fee2e2; color:#991b1b; }
  .answer  { background:#f8fafc; border-left:4px solid #2563eb;
             border-radius:6px; padding:14px 18px; margin-top:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">✈️ Airline Customer Support</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Powered by LangChain · LangGraph · Pinecone · PostgreSQL</p>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("What can I help with?")
    st.markdown("""
- ✈️ Flight status & delays
- 💺 Seat availability & fares
- 🚪 Gate & terminal info
- 🧳 Baggage policies
- 🔄 Cancellation & refunds
- ♿ Special assistance
- 📋 Booking policies
    """)
    st.divider()
    st.subheader("Sample queries")
    samples = [
        "What is the status of flight 6E477 on 10 Nov 2026?",
        "Are there flights from Delhi to Nagpur on 11 Nov 2026?",
        "Show available flights from Mumbai to Bengaluru under 6000 INR.",
        "How much free baggage is allowed for domestic flights?",
        "What is the cancellation policy?",
        "Can I carry a power bank in cabin baggage?",
    ]
    chosen = st.selectbox("Try a sample:", [""] + samples)

# ── Main input ─────────────────────────────────────────────────────────────
user_query = st.text_area(
    "Your question:",
    value=chosen,
    height=90,
    placeholder="e.g. What is the status of flight 6E477 on 10 Nov 2026?",
)

_, btn_col, _ = st.columns([1, 2, 1])
submit = btn_col.button("🔍 Ask", use_container_width=True, type="primary")

# ── Handle submission ──────────────────────────────────────────────────────
if submit:
    if not user_query.strip():
        st.warning("Please enter a question before clicking Ask.")
    else:
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(API_URL, json={"query": user_query}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except requests.ConnectionError:
                st.error("Cannot reach the API. Make sure FastAPI is running on port 8000.")
                st.stop()
            except requests.Timeout:
                st.error("Request timed out. Try again.")
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.stop()

        st.divider()

        # Category badge
        cat = data.get("category", "")
        badge_class = {"Need SQL": "sql", "Non SQL": "rag",
                       "Out of Context": "ooc", "BLOCKED": "blocked"}.get(cat, "ooc")
        badge_label = {"Need SQL": "✈️ Live Flight Data", "Non SQL": "📚 Knowledge Base",
                       "Out of Context": "🌐 Out of Scope", "BLOCKED": "🚫 Blocked"}.get(cat, cat)
        st.markdown(f'<span class="chip {badge_class}">{badge_label}</span>', unsafe_allow_html=True)

        # Generated SQL (if any)
        if data.get("sql"):
            with st.expander("Generated SQL"):
                st.code(data["sql"], language="sql")

        # Answer
        st.markdown(f'<div class="answer">{data["response"]}</div>', unsafe_allow_html=True)
