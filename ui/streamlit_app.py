import os
import streamlit as st

from engine.pipeline import generate_for_customer

# ---------- BASIC PAGE CONFIG ----------
st.set_page_config(
    page_title="Union Bank GenAI Personalization",
    layout="wide",
    page_icon="💬",
)

# ---------- GLOBAL CSS ----------
BASE_CSS = """
<style>
/* Hide Streamlit default menu & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Make chat wider & cleaner */
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.8rem;
}

/* Chat bubble width */
[data-testid="stChatMessage"] {
    max-width: 900px;
}

/* Header styles */
.header-container {
    display: flex;
    align-items: flex-start;
    border-bottom: 2px solid #B40001;  /* Union Bank red */
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}

.header-title {
    font-size: 1.6rem;
    font-weight: 900;
    color: #B40001;
    letter-spacing: 0.6px;
}

.header-subtitle {
    font-size: 0.9rem;
    margin-top: 0.1rem;
    color: #444444;
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGO_PATH = os.path.join(PROJECT_ROOT, "ui", "union_bank_logo.png")

# ---------- MAIN PAGE HEADER (NO LOGO, ONLY TITLE) ----------
HEADER_HTML = """
<div class="header-container">
    <div>
        <div class="header-title">UNION BANK OF INDIA</div>
        <div class="header-subtitle">
            GenAI Content Personalization & Customer Engagement Bot
        </div>
    </div>
</div>
"""
st.markdown(HEADER_HTML, unsafe_allow_html=True)

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
    st.session_state.use_llm = False  # start with dummy variants


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


# ---------- SIDEBAR (LOGO + SETTINGS + COMPACT DASHBOARD) ----------
with st.sidebar:

    # --- BIG UNION BANK LOGO AT TOP LEFT ---
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=180)  # big logo
    else:
        st.markdown("## 🏦")

    st.markdown("---")

    # ---- CUSTOMER SETTINGS ----
    st.header("📇 Customer & Settings")

    st.session_state.customer_id = st.text_input(
        "Customer ID",
        value=st.session_state.customer_id,
        help="Use IDs like C00001, C00002, etc. from synthetic data.",
    )

    st.session_state.use_llm = st.checkbox(
        "Use Ollama LLM (if available)",
        value=st.session_state.use_llm,
        help="Keep OFF to use dummy variants (no Ollama needed).",
    )

    st.markdown("---")

    # ---- COMPACT CUSTOMER 360 DASHBOARD IN SIDEBAR ----
    st.subheader("🧩 Customer 360")

    dashboard_data, dashboard_err = run_personalization(
        st.session_state.customer_id,
        use_llm=False,  # always use fast dummy variants for overview
    )

    if dashboard_err:
        st.caption(f"Could not load overview: {dashboard_err}")
    else:
        cust = dashboard_data["customer"]
        prod = dashboard_data["product"]

        # Basic profile
        st.markdown(f"**{cust['name']}**  \n`{cust['customer_id']}`")
        st.caption(
            f"{cust['city']} · Age {cust['age']} · {cust['lifecycle_stage']}"
        )

        # Scores (compact)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Engagement", f"{cust['engagement_score']:.1f}")
            st.metric("Value score", f"{cust['value_score']:.2f}")
        with col_b:
            st.metric("Avg balance", f"₹{cust['avg_monthly_balance']:,.0f}")
            st.caption(f"Risk: `{cust['risk_profile']}`")

        st.markdown("---")

        # Recommended product (compact)
        st.markdown("🎯 **Recommended Product**")
        st.markdown(f"**{prod['name']}**  \n`{prod['category']}`")

        for b in prod.get("benefits", [])[:3]:
            st.write(f"- {b}")

    st.markdown("---")
    st.caption("This panel shows live customer context & recommendation.")


# ---------- MAIN CHAT AREA ----------
st.markdown("### 💬 Campaign Chat & Message Generation")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input at bottom
user_input = st.chat_input(
    "Describe your campaign goal, or just press Enter to generate messages."
)

if user_input is not None:
    user_input = user_input.strip()

if user_input:
    # 1) Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Call pipeline (here we respect the 'use_llm' toggle)
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
            st.session_state.messages.append(
                {"role": "assistant", "content": reply_text}
            )
