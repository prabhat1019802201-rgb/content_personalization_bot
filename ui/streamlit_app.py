import os
import streamlit as st

from engine.pipeline import generate_for_customer

# ---------- BASIC PAGE CONFIG ----------
st.set_page_config(
    page_title="Union Bank GenAI Personalization",
    layout="wide",
    page_icon="💬",
)

# ---------- CUSTOM CSS TO LOOK CHAT-LIKE ----------
CHAT_CSS = """
<style>
/* Hide Streamlit default menu & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Top brand bar */
.brand-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
}
.brand-title {
    font-weight: 700;
    font-size: 1.1rem;
}
.brand-subtitle {
    font-size: 0.8rem;
    color: #666;
}

/* Make chat wider & cleaner */
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.8rem;
}

/* Chat bubble width */
[data-testid="stChatMessage"] {
    max-width: 900px;
}
</style>
"""
st.markdown(CHAT_CSS, unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Namaste! I’m your **Union Bank GenAI personalization assistant**.\n\n"
                "Enter a customer ID on the left, then send a message here to generate "
                "personalized A/B/C content and see the best variant."
            ),
        }
    ]

if "customer_id" not in st.session_state:
    st.session_state.customer_id = "C00001"

if "use_llm" not in st.session_state:
    # For now keep False (dummy variants) – you can switch later once Ollama is ready.
    st.session_state.use_llm = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGO_PATH = os.path.join(PROJECT_ROOT, "ui", "union_bank_logo.png")


# ---------- HEADER WITH LOGO + TITLE ----------
with st.container():
    cols = st.columns([0.08, 0.92])
    with cols[0]:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=48)
        else:
            st.write("🏦")
    with cols[1]:
        st.markdown(
            """
            <div class="brand-bar">
                <div>
                    <div class="brand-title">Union Bank of India – GenAI Personalization</div>
                    <div class="brand-subtitle">
                        Content Personalization & Generation Bot for Customer Engagement
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("📇 Customer & Settings")

    st.session_state.customer_id = st.text_input(
        "Customer ID",
        value=st.session_state.customer_id,
        help="Use IDs like C00001, C00002, etc. from synthetic data.",
    )

    st.session_state.use_llm = st.checkbox(
        "Use Ollama LLM (if available)",
        value=st.session_state.use_llm,
        help=(
            "Keep OFF to use dummy variants (no Ollama needed).\n"
            "Turn ON only after you have Ollama + model configured."
        ),
    )

    st.markdown("---")
    st.caption(
        "Tip: this UI calls the Python pipeline directly, no separate API server needed."
    )


# ---------- HELPER TO CALL BACKEND PIPELINE ----------
def run_personalization(customer_id: str, use_llm: bool):
    """
    Call our engine.pipeline and handle errors cleanly.
    """
    try:
        result = generate_for_customer(customer_id, use_llm=use_llm)
        return result, None
    except Exception as e:
        return None, str(e)


# ---------- RENDER CHAT HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- CHAT INPUT (BOTTOM, LIKE CHATGPT) ----------
user_input = st.chat_input("Describe your campaign goal, or just press Enter to generate messages.")

if user_input is not None:
    user_input = user_input.strip()

if user_input:
    # 1) Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Call pipeline
    with st.chat_message("assistant"):
        with st.spinner("Generating personalized content..."):
            data, err = run_personalization(
                st.session_state.customer_id,
                use_llm=st.session_state.use_llm,
            )

        if err is not None:
            st.error(f"❌ Error: {err}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"❌ Error: {err}"}
            )
        else:
            cust = data["customer"]
            prod = data["product"]
            selected = data["selected"]
            variants_scored = data["variants_scored"]

            # Build a readable reply
            lines = []

            lines.append(
                f"📨 **Personalized message for {cust['name']}**  \n"
                f"- Customer ID: `{cust['customer_id']}`  \n"
                f"- Segment: `{cust['segment']}`  \n"
                f"- Preferred language: `{cust['preferred_language']}`  \n"
                f"- Primary channel: `{cust['primary_channel']}`  \n"
                f"- Product: **{prod['name']}** ({prod['category']})  \n"
            )

            if selected:
                lines.append("\n### ✅ Selected Variant")
                lines.append(f"**Variant:** `{selected.get('variant_tag', 'N/A')}`  ")

                body = (selected.get("body") or "").strip()
                subject = (selected.get("subject") or "").strip()
                disclaimers = selected.get("disclaimers") or []

                if cust["primary_channel"] == "sms":
                    lines.append("**SMS Text:**")
                    lines.append(f"> {body}")
                elif cust["primary_channel"] == "email":
                    lines.append(f"**Subject:** {subject}")
                    lines.append("**Body:**")
                    lines.append(f"> {body}")
                else:  # app
                    lines.append(f"**App Banner Title:** {subject}")
                    lines.append("**App Banner Text:**")
                    lines.append(f"> {body[:200]}")

                if disclaimers:
                    lines.append("\n**Disclaimers:**")
                    for d in disclaimers:
                        lines.append(f"- {d}")

            # Show all variants with scores
            if variants_scored:
                lines.append("\n---\n")
                lines.append("### 🅰🅱🅲 All Variants & Scores\n")
                for item in variants_scored:
                    score = item["score"]
                    v = item["variant"]
                    tag = v.get("variant_tag", "?")
                    compliant = v.get("compliant", True)
                    rejected = v.get("rejected_by_policy", False)
                    reason = v.get("rejection_reason", "")

                    body = (v.get("body") or "").strip().replace("\n", "  \n")

                    status = "OK"
                    if rejected:
                        status = f"REJECTED ({reason})"
                    elif not compliant:
                        status = "NON-COMPLIANT"

                    lines.append(
                        f"**Variant {tag}** — Score: `{round(score, 2)}` | Status: `{status}`  \n"
                        f"> {body}\n"
                    )

            reply_text = "\n".join(lines)
            st.markdown(reply_text)
            st.session_state.messages.append({"role": "assistant", "content": reply_text})
