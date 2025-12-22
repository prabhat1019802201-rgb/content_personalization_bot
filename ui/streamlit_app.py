# ui/streamlit_app.py
import os
import json
import base64
import streamlit as st
import pandas as pd

from engine.pipeline import generate_for_customer, detect_intent
from engine.campaigns import get_top_customers_for_product
from engine.pipeline import recommend_top_products

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Union Bank GenAI Personalization",
    layout="wide",
    page_icon="🏦",
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stChatMessage"] {max-width: 900px;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGO_PATH = os.path.join(PROJECT_ROOT, "ui", "union_bank_logo.png")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "creative_config.json")
CUSTOMERS_CSV = os.path.join(PROJECT_ROOT, "data", "customers.csv")
PRODUCTS_CSV = os.path.join(PROJECT_ROOT, "data", "products.csv")

# =============================================================================
# SESSION STATE (CRITICAL – DO NOT REMOVE)
# =============================================================================
st.session_state.setdefault("customer_id", "C00001")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("creative_kit", {"banner": True, "whatsapp": True, "email": True})
st.session_state.setdefault("text_model_choice", "Default (rule-based)")
st.session_state.setdefault("image_model_choice", "Default (stub)")
st.session_state.setdefault("custom_ollama_model", "")

st.session_state.setdefault("campaign_list", [])
st.session_state.setdefault("campaign_page", 0)

# =============================================================================
# SAFE BACKEND CALL
# =============================================================================
def safe_generate(customer_id, **kwargs):
    res = generate_for_customer(customer_id, **kwargs)
    if isinstance(res, tuple):
        data, err = res
        if err:
            raise RuntimeError(err)
        return data
    return res if isinstance(res, dict) else {}

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
### Union Bank of India – GenAI Personalization  
Customer Engagement • Campaign Targeting • Creative Intelligence
""")
st.markdown("---")

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:

    # Logo
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"<div style='text-align:center'><img src='data:image/png;base64,{b64}' width='220'/></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.header("📇 Customer Settings")

    st.session_state.customer_id = st.text_input(
        "Customer ID",
        value=st.session_state.customer_id
    )

    # ---------------- Customer 360 ----------------
    st.markdown("---")
    st.subheader("🔎 Customer 360")

    try:
        rec_data = safe_generate(st.session_state.customer_id, use_llm=False)
    except Exception as e:
        rec_data = {}
        st.warning(f"Customer load failed: {e}")

    customer = rec_data.get("customer", {}) if isinstance(rec_data, dict) else {}

    if customer:
        st.markdown(f"""
**{customer.get("name","Unknown")}**  
ID: `{customer.get("customer_id","-")}`  
Age: {customer.get("age","-")}  
City: {customer.get("city","-")}  
Channel: {customer.get("primary_channel","-")}
""")

        st.caption(
            f"Lifecycle: {customer.get('lifecycle_stage','-')} • "
            f"Risk: {customer.get('risk_profile','-')}"
        )

        cols = st.columns(2)
        cols[0].metric("Avg Balance", f"₹{customer.get('avg_monthly_balance','-')}")
        cols[1].metric("Tenure (months)", customer.get("relationship_tenure_months","-"))
    else:
        st.info("Customer not found")

    # ---------------- Recommended Product ----------------
    st.markdown("---")
    st.subheader("🔍 Recommended Products")

    recommended = rec_data.get("recommended_products", [])

    if recommended:
      for idx, p in enumerate(recommended, start=1):
           name = p.get("name", "")
           category = p.get("category", "")
           badge = " ⭐" if idx == 1 else ""

           st.markdown(f"""
          **{idx}. {name}{badge}**  
          Category: {category}
       """)
    else:
       st.caption("No product recommendations available")

    # ---------------- Advanced Config ----------------
    st.markdown("---")
    st.subheader("⚙️ Advanced Options")

    with st.expander("Model & Creative Settings"):

        st.selectbox(
            "Text model",
            ["Default (rule-based)", "qwen2.5vl:7b", "llama3.1:8b"],
            key="text_model_choice"
        )

        st.text_input("Custom Ollama model", key="custom_ollama_model")

        st.selectbox(
            "Image model",
            ["Default (stub)", "Automatic1111", "Diffusers SDXL", "No image"],
            key="image_model_choice"
        )

        ck = st.session_state.creative_kit
        st.session_state.creative_kit = {
            "banner": st.checkbox("Banner", ck["banner"]),
            "whatsapp": st.checkbox("WhatsApp", ck["whatsapp"]),
            "email": st.checkbox("Email", ck["email"]),
        }

        if st.button("💾 Save configuration"):
            with open(CONFIG_PATH, "w") as f:
                json.dump({
                    "text_model": st.session_state.text_model_choice,
                    "image_model": st.session_state.image_model_choice,
                    "creative_kit": st.session_state.creative_kit
                }, f, indent=2)
            st.success("Saved")

# =============================================================================
# MAIN TABS
# =============================================================================
tab_chat, tab_campaign = st.tabs(["💬 Chat Mode", "📢 Campaign Mode"])

# =============================================================================
# CHAT MODE
# =============================================================================
with tab_chat:

    # 1️⃣ Render entire chat history FIRST
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 2️⃣ Chat input MUST be last (Streamlit rule)
    user_input = st.chat_input("Type a message and press Enter...")

    # 3️⃣ Handle input AFTER rendering history
    if user_input:
        # Save user message immediately
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        intent = detect_intent(user_input)

        # Prepare assistant response
        reply = ""

        with st.chat_message("assistant"):

            # ---------- GENERIC CHAT ----------
            if intent == "GENERIC_CHAT":
                reply = (
                    "👋 Hello! I can help you with:\n\n"
                    "- Customer profile & insights\n"
                    "- Product recommendations\n"
                    "- Personalized marketing messages\n\n"
                    f"Try asking:\n"
                    f"• *Tell me about {st.session_state.customer_id}*\n"
                    f"• *What product can be sold to this user?*\n"
                    f"• *Generate marketing content for this customer*"
                )
                st.markdown(reply)

            # ---------- CUSTOMER INFO ----------
            elif intent == "CUSTOMER_INFO":
                with st.spinner("Fetching customer details..."):
                    data = generate_for_customer(
                        st.session_state.customer_id,
                        use_llm=False
                    )

                cust = data.get("customer", {})
                metrics = data.get("metrics", {})

                reply = f"""
### 👤 Customer 360 Overview

**Name:** {cust.get('name')}
**Customer ID:** `{cust.get('customer_id')}`
**City:** {cust.get('city')}
**Lifecycle:** {cust.get('lifecycle_stage')}
**Risk Profile:** {cust.get('risk_profile')}

**Avg Monthly Balance:** ₹{cust.get('avg_monthly_balance')}
**Engagement Score:** {metrics.get('engagement_score')}
"""
                st.markdown(reply)

            # ---------- PRODUCT ADVISORY ----------
            elif intent == "PRODUCT_ADVISORY":
                with st.spinner("Analyzing customer and recommending products..."):
                    data = generate_for_customer(
                        st.session_state.customer_id,
                        use_llm=False
                    )

                cust = data.get("customer", {})
                products = data.get("recommended_products", [])

                if not products:
                    reply = "No suitable products found for this customer."
                else:
                    reply = f"""
### 🧠 Product Advisory

Based on **{cust.get('name')}**’s profile, the following products can be offered:
"""
                    for i, p in enumerate(products, 1):
                        reply += f"""
**{i}. {p.get('name')}**
- Category: {p.get('category')}
"""

                st.markdown(reply)

            # ---------- MARKETING GENERATION ----------
            elif intent == "MARKETING_GEN":
                with st.spinner("Generating personalized marketing content..."):
                    data = generate_for_customer(
                        st.session_state.customer_id,
                        user_request=user_input,
                        use_llm=True,
                        text_model_choice=st.session_state.text_model_choice,
                        image_model_choice=st.session_state.image_model_choice,
                        creative_kit=st.session_state.creative_kit,
                    )

                cust = data.get("customer", {})
                prod = data.get("product", {})
                selected = data.get("selected", {})

                reply = f"""
### 📨 Personalized Marketing Content

**Customer:** {cust.get('name')}
**Product:** {prod.get('name')}
"""

                for channel, variant in selected.items():
                    if not variant:
                        continue
                    reply += f"\n#### {channel.upper()}\n"
                    if variant.get("subject"):
                        reply += f"**Subject:** {variant.get('subject')}\n"
                    reply += f"{variant.get('body')}\n"

                st.markdown(reply)

            else:
                reply = "Sorry, I didn’t understand that."
                st.markdown(reply)

        # 4️⃣ Persist assistant reply LAST
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply
        })

        # 5️⃣ Force rerun → keeps input fixed at bottom
        st.rerun()
# =============================================================================
# CAMPAIGN MODE TAB — CSV STYLE WITH PAGINATION
# =============================================================================
with tab_campaign:
    st.header("📢 Campaign Mode – Target Customers")

    # ---------- Product selection ----------
    product_id = st.text_input(
        "Product ID",
        value=st.session_state.get("campaign_product_id", "P001"),
        help="Example: P001, P007 etc."
    )

    # ---------- Fetch customers ----------
    if st.button("Fetch Target Customers"):
        with st.spinner("Fetching targeted customers..."):
            try:
                cust_ids = get_top_customers_for_product(product_id)

                if not cust_ids:
                    st.warning("No customers found for this product.")
                else:
                    st.session_state.campaign_product_id = product_id
                    st.session_state.campaign_list = list(map(str, cust_ids))
                    st.session_state.campaign_page = 0
                    st.success(f"Found {len(cust_ids)} customers")

            except Exception as e:
                st.error(f"Error fetching customers: {e}")

    # ---------- Show table with pagination ----------
    if "campaign_list" in st.session_state and st.session_state.campaign_list:

        PAGE_SIZE = 20

        customers = st.session_state.campaign_list
        page = st.session_state.campaign_page
        total_customers = len(customers)
        total_pages = (total_customers + PAGE_SIZE - 1) // PAGE_SIZE

        start = page * PAGE_SIZE
        end = start + PAGE_SIZE
        page_customers = customers[start:end]

        st.markdown(
            f"### Showing customers {start + 1} – {min(end, len(customers))} "
            f"of {len(customers)}"
        )

        # ---------- Load customer CSV for enrichment ----------
        import pandas as pd
        cust_csv = os.path.join(PROJECT_ROOT, "data", "customers.csv")

        if os.path.exists(cust_csv):
            cdf = pd.read_csv(cust_csv, dtype=str)
            rows = []

            for cid in page_customers:
                row = cdf[cdf["customer_id"] == cid]
                if not row.empty:
                    r = row.iloc[0]
                    rows.append({
                        "Customer ID": cid,
                        "Name": r.get("name", ""),
                        "City": r.get("city", ""),
                        "Language": r.get("preferred_language", ""),
                        "Lifecycle": r.get("lifecycle_stage", ""),
                        "Risk": r.get("risk_profile", "")
                    })
                else:
                    rows.append({"Customer ID": cid})

            df_show = pd.DataFrame(rows)
            st.dataframe(df_show, use_container_width=True)

        else:
            # fallback: show IDs only
            st.dataframe(
                pd.DataFrame({"Customer ID": page_customers}),
                use_container_width=True
            )

        # ---------- Pagination controls ----------
        col_prev, col_next = st.columns([1, 1])

        with col_prev:
            if st.button("⬅ Previous", disabled=(page <= 0)):
              st.session_state.campaign_page -= 1
              st.rerun()

        with col_next:
            if st.button("Next ➡", disabled=(page >= total_pages)):
            ##if st.button("Next ➡", disabled=False):
               st.session_state.campaign_page += 1
               st.rerun()

