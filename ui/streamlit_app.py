# ui/streamlit_app.py
import os
import json
import streamlit as st

from engine.pipeline import generate_for_customer
from engine.campaigns import get_top_customers_for_product

# =============================================================================
# BASIC PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Union Bank GenAI Personalization",
    layout="wide",
    page_icon="🏦",
)

# =============================================================================
# CUSTOM CSS TO LOOK LIKE CHATGPT
# =============================================================================
CHAT_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.brand-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.3rem 0;
}
.brand-title {
    font-weight: 700;
    font-size: 1.2rem;
}
.brand-subtitle {
    font-size: 0.8rem;
    color: #777;
}
[data-testid="stChatMessage"] {
    max-width: 900px;
}
</style>
"""
st.markdown(CHAT_CSS, unsafe_allow_html=True)

# =============================================================================
# LOAD PATHS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGO_PATH = os.path.join(PROJECT_ROOT, "ui", "union_bank_logo.png")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "creative_config.json")

# =============================================================================
# SESSION STATE SETUP
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Namaste! I’m your **Union Bank GenAI personalization assistant**.\n\n"
                "Enter a customer ID on the left, then send a message here to generate "
                "personalized A/B/C content and creatives."
            ),
        }
    ]

if "customer_id" not in st.session_state:
    st.session_state.customer_id = "C00001"

if "campaign_page" not in st.session_state:
    st.session_state.campaign_page = 0

# =============================================================================
# HEADER (MAIN PAGE TOP)
# =============================================================================
with st.container():
    # keep minimal top header (logo shown in sidebar as requested)
    st.markdown("""
        <div style="padding:4px 0;">
            <div style="font-weight:700;font-size:1.1rem;">Union Bank of India – GenAI Personalization</div>
            <div style="font-size:0.85rem;color:#666;">Customer Engagement • Campaign Personalization • Creative Generation</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ------------------- Safe wrapper for generate_for_customer -------------------
def _call_generate_for_customer_safe(customer_id: str, **kwargs):
    """
    Call engine.generate_for_customer and normalize return values.

    Accepts backends that return either:
     - a dict (data), or
     - a tuple (data, err)

    On error, raises the underlying exception so the UI can catch & display it.
    """
    try:
        res = generate_for_customer(customer_id, **kwargs)
    except Exception as e:
        # propagate exception to be handled by caller (sidebar)
        raise

    # If backend returns (data, err)
    if isinstance(res, tuple):
        if len(res) == 2:
            data, err = res
            if err:
                raise RuntimeError(err)
            return data
        # unexpected shape — return first element if dict-like
        try:
            return res[0]
        except Exception:
            raise RuntimeError("generate_for_customer returned unexpected tuple shape")

    # If backend returned dict-like, return as-is
    return res
# ---------------------------------------------------------------------------

# =============================================================================
# SIDEBAR — logo on top + CUSTOMER + RECOMMENDED PRODUCT + ADVANCED OPTIONS
# =============================================================================
with st.sidebar:

    # --- Logo at top of sidebar (user requested) ---
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=96)
    else:
        st.markdown("### 🏦 Union Bank of India")
    st.markdown("---")

    st.header("📇 Customer Settings")
    st.session_state.customer_id = st.text_input(
        "Customer ID",
        value=st.session_state.customer_id,
        help="Try C00001, C00002, etc.",
    )

    st.markdown("---")
    st.subheader("🔎 Customer 360 (Overview)")

    # Safe wrapper already present: _call_generate_for_customer_safe(customer_id, **kwargs)
    try:
        rec_data = _call_generate_for_customer_safe(st.session_state.customer_id, use_llm=False)
    except Exception as e:
        rec_data = None
        st.warning(f"Could not load customer overview: {e}")

    if rec_data and isinstance(rec_data, dict):
        cust = rec_data.get("customer", {})

        # Basic profile
        if cust:
            st.markdown(f"**{cust.get('name','Unknown')}**  \n"
                        f"- Customer ID: `{cust.get('customer_id','-')}`  \n"
                        f"- Age: {cust.get('age','-')}  \n"
                        f"- City: {cust.get('city','-')}  \n"
                        f"- Preferred language: {cust.get('preferred_language','-')}  \n"
                        f"- Primary channel: {cust.get('primary_channel','-')}")
            st.caption(f"Lifecycle: {cust.get('lifecycle_stage','-')}  •  Risk: {cust.get('risk_profile','-')}")
        else:
            st.info("No customer profile available.")

        # Key numeric KPIs
        avg_balance = cust.get("avg_monthly_balance") or rec_data.get("metrics", {}).get("avg_monthly_balance")
        tenure = cust.get("relationship_tenure_months") or rec_data.get("metrics", {}).get("relationship_tenure_months")
        engagement = rec_data.get("metrics", {}).get("engagement_score")
        value_score = rec_data.get("metrics", {}).get("value_score")

        cols_kpi = st.columns(4)
        with cols_kpi[0]:
            try:
                st.metric("Avg Monthly Balance", f"₹{int(avg_balance):,}" if avg_balance is not None else "-")
            except Exception:
                st.metric("Avg Monthly Balance", f"{avg_balance}" if avg_balance is not None else "-")
        with cols_kpi[1]:
            st.metric("Tenure (months)", f"{int(tenure)}" if tenure is not None else "-")
        with cols_kpi[2]:
            st.metric("Engagement", f"{round(engagement,2)}" if engagement is not None else "-")
        with cols_kpi[3]:
            st.metric("Value score", f"{round(value_score,2)}" if value_score is not None else "-")

        # Product / Holdings info
        st.markdown("**Holdings / Status**")
        cc = cust.get("credit_card_holder")
        loan = cust.get("loan_holder") or rec_data.get("product_holdings", {}).get("loan_holder")
        recent_issues = cust.get("recent_issues") or rec_data.get("recent_issues") or []
        st.markdown(f"- Credit Card Holder: `{cc}`")
        st.markdown(f"- Loan Holder: `{loan}`")
        if recent_issues:
            if isinstance(recent_issues, (list, tuple)):
                st.markdown("- Recent issues:")
                for it in recent_issues:
                    st.markdown(f"  - {it}")
            else:
                st.markdown(f"- Recent issues: `{recent_issues}`")
        else:
            st.markdown("- Recent issues: `none`")

        # Recent events
        events = rec_data.get("recent_events") or rec_data.get("events") or []
        if events:
            st.markdown("**Recent events (latest 5)**")
            for ev in events[:5]:
                ts = ev.get("event_ts") or ev.get("ts") or ""
                etype = ev.get("event_type") or ev.get("type") or ""
                amt = ev.get("amount")
                ch = ev.get("channel") or ""
                line = f"- {ts} • {etype}"
                if amt is not None and str(amt).strip() != "":
                    line += f" • ₹{amt}"
                if ch:
                    line += f" • {ch}"
                st.markdown(line)
        else:
            st.caption("No recent events available.")

        # Optional raw debug
        with st.expander("More details (raw)", expanded=False):
            st.json(rec_data)

    else:
        st.info("Customer 360 not available for this ID.")

    # Try loading recommended product via pipeline (safe)
    try:
        rec_data = _call_generate_for_customer_safe(st.session_state.customer_id, use_llm=False)
    except Exception as e:
        rec_data = None
        st.warning(f"Could not load customer overview: {e}")

    if rec_data and isinstance(rec_data, dict):
        # show profile
        cust = rec_data.get("customer", {})
        if cust:
            st.markdown(f"**{cust.get('name','Unknown')}**  \n"
                        f"- Customer ID: `{cust.get('customer_id','-')}`  \n"
                        f"- Age: {cust.get('age','-')}  \n"
                        f"- City: {cust.get('city','-')}  \n"
                        f"- Primary channel: {cust.get('primary_channel','-')}")
            st.caption(f"Lifecycle: {cust.get('lifecycle_stage','-')}  •  Risk: {cust.get('risk_profile','-')}")
        else:
            st.info("No customer profile available.")

        # show small metrics if available
        engagement = rec_data.get("metrics", {}).get("engagement_score")
        value_score = rec_data.get("metrics", {}).get("value_score")
        if engagement is not None or value_score is not None:
            cols_m = st.columns(2)
            with cols_m[0]:
                st.metric("Engagement", f"{engagement if engagement is not None else '-'}")
            with cols_m[1]:
                st.metric("Value", f"{value_score if value_score is not None else '-'}")

        # show recent events (if pipeline returned)
        events = rec_data.get("recent_events") or rec_data.get("events") or []
        if events:
            st.markdown("**Recent events (latest 5)**")
            for ev in events[:5]:
                ts = ev.get("event_ts") or ev.get("ts") or ""
                etype = ev.get("event_type") or ev.get("type") or ""
                amt = ev.get("amount")
                ch = ev.get("channel") or ""
                line = f"- {ts} • {etype}"
                if amt:
                    line += f" • ₹{amt}"
                if ch:
                    line += f" • {ch}"
                st.markdown(line)
        else:
            st.caption("No recent events available.")
    else:
        st.info("Customer 360 not available for this ID.")

    st.markdown("---")
    st.subheader("🔍 Recommended Product")

    try:
        if rec_data:
            prod = rec_data.get("product", {})
            if prod:
                st.markdown(f"**{prod.get('name','N/A')}**")
                st.caption(f"Category: {prod.get('category','N/A')}")
                targets = prod.get("cross_sell_targets", [])
                if isinstance(targets, (list, tuple)) and targets:
                    st.caption(f"Targeted: {', '.join(targets)}")
            else:
                st.info("No product recommendation available.")
        else:
            st.info("No product recommendation available.")
    except Exception as e:
        st.warning(f"Could not load product overview: {e}")

    st.markdown("---")
    st.subheader("⚙️ Advanced Generation Options")

    with st.expander("Model & Creative Configuration", expanded=False):

        # ---------------- TEXT MODEL ----------------
        st.markdown("### 🧠 Text Model")
        text_model_opt = st.selectbox(
            "Choose text model",
            [
                "Default (rule-based)",
                "qwen2.5vl:7b",
                "llama3.1:8b",
                "Custom Ollama model",
            ],
        )
        st.session_state.text_model_choice = text_model_opt

        st.session_state.custom_ollama_model = st.text_input(
            "If custom, enter model name",
            value=st.session_state.get("custom_ollama_model", ""),
        )

        # ---------------- IMAGE MODEL ----------------
        st.markdown("### 🖼 Image Model")
        image_model_opt = st.selectbox(
            "Choose image model",
            [
                "Default (stub)",
                "Automatic1111 (http://localhost:7860)",
                "Diffusers SDXL (local GPU)",
                "No image (text only)",
            ],
        )
        st.session_state.image_model_choice = image_model_opt

        # ---------------- CREATIVE KIT ----------------
        st.markdown("### 🎨 Creative Kit")
        ck = st.session_state.get("creative_kit", {"banner": True, "whatsapp": True, "email": True})
        cols_ck = st.columns(3)
        with cols_ck[0]:
            ck["banner"] = st.checkbox("Banner", value=ck.get("banner", True))
        with cols_ck[1]:
            ck["whatsapp"] = st.checkbox("WhatsApp", value=ck.get("whatsapp", True))
        with cols_ck[2]:
            ck["email"] = st.checkbox("Email", value=ck.get("email", True))
        st.session_state.creative_kit = ck

        # ---------------- IMAGE GENERATION SETTINGS ----------------
        st.markdown("### 📐 Image Generation Settings")

        st.session_state.creative_width = st.number_input(
            "Width", 600, 2000, int(st.session_state.get("creative_width", 1200))
        )
        st.session_state.creative_height = st.number_input(
            "Height", 200, 1200, int(st.session_state.get("creative_height", 400))
        )
        st.session_state.creative_steps = st.number_input(
            "Steps", 5, 150, int(st.session_state.get("creative_steps", 20))
        )
        st.session_state.creative_device = st.text_input(
            "Device (cuda / cpu)",
            value=st.session_state.get("creative_device", "cuda"),
        )

        # ---------------- SAVE CONFIG ----------------
        if st.button("💾 Save All Configuration"):
            cfg = {
                "text_model_choice": st.session_state.text_model_choice,
                "custom_ollama_model": st.session_state.custom_ollama_model,
                "image_model_choice": st.session_state.image_model_choice,
                "creative_kit": st.session_state.creative_kit,
                "width": st.session_state.creative_width,
                "height": st.session_state.creative_height,
                "steps": st.session_state.creative_steps,
                "device": st.session_state.creative_device,
            }
            try:
                with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                    json.dump(cfg, fh, indent=2)
                st.success("Configuration saved!")
            except Exception as e:
                st.error(f"Failed to save config: {e}")

# =============================================================================
# MAIN BODY — TABS (CHAT MODE / CAMPAIGN MODE)
# =============================================================================
tab_chat, tab_campaign = st.tabs(["💬 Chat Mode", "📢 Campaign Mode"])

# =============================================================================
# CHAT MODE TAB
# =============================================================================
with tab_chat:

    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Type a message or just press Enter to generate messages...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating personalized output..."):
                # call pipeline and normalize return as (data, err) style if needed
                try:
                    res = generate_for_customer(
                        st.session_state.customer_id,
                        text_model_choice=st.session_state.get("text_model_choice"),
                        custom_model=st.session_state.get("custom_ollama_model"),
                        image_model_choice=st.session_state.get("image_model_choice"),
                        creative_kit=st.session_state.get("creative_kit", {"banner": True, "whatsapp": True, "email": True}),
                        creative_width=st.session_state.get("creative_width", 1200),
                        creative_height=st.session_state.get("creative_height", 400),
                        creative_steps=st.session_state.get("creative_steps", 20),
                    )
                    # normalize
                    if isinstance(res, tuple) and len(res) == 2:
                        data, err = res
                    else:
                        data = res
                        err = None
                except Exception as e:
                    data = None
                    err = str(e)

            if err:
                st.error(f"❌ Error: {err}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ Error: {err}"}
                )

            else:
                cust = data.get("customer", {})
                prod = data.get("product", {})
                selected = data.get("selected", {})
                variants_scored = data.get("variants_scored", [])
                creative_result = data.get("creative_result")

                # ---------------- TEXT CONTENT OUTPUT ----------------
                st.markdown(
                    f"""
                    ### 📨 Personalized Message for **{cust.get('name','Unknown')}**
                    - **Customer ID:** `{cust.get('customer_id','-')}`
                    - **Segment:** `{cust.get('segment','-')}`
                    - **Product:** **{prod.get('name','-')}** ({prod.get('category','-')})
                    """
                )

                # Selected Variant
                st.markdown("### ✅ Selected Variant")
                st.markdown(f"**Variant:** `{selected.get('variant_tag','?')}`")
                body_text = selected.get('body') or selected.get('message') or ""
                st.markdown(f"> {body_text.strip()}")

                # ---------------- CREATIVE DISPLAY ----------------
                if creative_result and st.session_state.creative_kit.get("banner", True):

                    st.markdown("### 🎨 Banner Creative Preview")

                    for item in creative_result.get("meta", {}).get("items", []):

                        for variant in item.get("variants", []):
                            size = variant.get("size")
                            fpath = variant.get("file")
                            review = variant.get("review", {})
                            score = review.get("score")
                            blocked = review.get("blocked")
                            comments = review.get("comments", [])

                            st.markdown(
                                f"**Size:** {size[0]}×{size[1]} | "
                                f"**Score:** `{score}` | "
                                f"**Blocked:** `{blocked}`"
                            )
                            try:
                                st.image(fpath, use_column_width=True)
                            except Exception as e:
                                st.warning(f"Unable to load image {fpath}: {e}")

                            if comments:
                                st.markdown("**Review Comments:**")
                                for c in comments:
                                    st.markdown(f"- {c}")

                            st.markdown("---")

                    # ZIP download
                    import zipfile, io
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                        for item in creative_result.get("meta", {}).get("items", []):
                            for variant in item.get("variants", []):
                                try:
                                    zipf.write(variant.get("file"), os.path.basename(variant.get("file")))
                                except Exception:
                                    pass
                        meta_path = os.path.join(creative_result.get("folder", ""), "meta.json")
                        if os.path.exists(meta_path):
                            try:
                                zipf.write(meta_path, "meta.json")
                            except Exception:
                                pass

                    st.download_button(
                        "📥 Download Full Creative Kit (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="creative_kit.zip",
                        mime="application/zip",
                    )

                # ---------------- VARIANTS A/B/C ----------------
                st.markdown("### 🅰🅱🅲 All Variants & Scores")

                for item in variants_scored:
                    v = item.get("variant", {})
                    score = item.get("score", 0.0)
                    tag = v.get("variant_tag", "?")
                    body = (v.get("body") or "").strip().replace("\n", "  \n")
                    st.markdown(
                        f"**Variant {tag}** — Score `{round(score, 2)}`\n> {body}\n"
                    )

                st.session_state.messages.append({"role": "assistant", "content": "Output generated."})


# =============================================================================
# CAMPAIGN MODE TAB (TEXT ONLY)
# =============================================================================
with tab_campaign:
    st.header("📢 Campaign Mode – Best 20 Customers for a Product")

    # Try to locate products.csv and build a product lookup
    products_csv = os.path.join(PROJECT_ROOT, "data", "products.csv")
    products_df = None
    product_options = []
    if os.path.exists(products_csv):
        try:
            import pandas as pd
            products_df = pd.read_csv(products_csv)
            # Expect columns: product_id, name (if different, adapt accordingly)
            # build options in "PID - Name" form
            for _, r in products_df.iterrows():
                pid = str(r.get("product_id") or r.get("id") or "")
                name = str(r.get("name") or r.get("product_name") or "")
                if pid and name:
                    product_options.append(f"{pid} — {name}")
        except Exception as e:
            st.warning(f"Could not load products.csv: {e}")

    # Render product selector (prefer selectbox if we have options)
    if product_options:
        selected_opt = st.selectbox("Pick product (id — name)", options=product_options)
        # extract product id
        product_id = selected_opt.split("—")[0].strip() if "—" in selected_opt else selected_opt.split("-")[0].strip()
    else:
        product_id = st.text_input("Product ID or Product Name (from products.csv)", value="P0001")

    if st.button("Fetch Target Customers"):
        try:
            # If user typed a name (not in pid format), try to resolve using the CSV
            pid_to_use = product_id
            if products_df is not None and not products_df.empty:
                # if input looks like name (no digits or not present as id), attempt name match
                if not any(products_df['product_id'].astype(str) == product_id):
                    # try match by name
                    matches = products_df[products_df.apply(
                        lambda r: str(r.get("name","")).strip().lower() == str(product_id).strip().lower() or
                                  str(r.get("product_name","")).strip().lower() == str(product_id).strip().lower()
                        , axis=1)]
                    if not matches.empty:
                        pid_to_use = str(matches.iloc[0].get("product_id"))
                    else:
                        # If product_id appears like "P0001 — Name", strip to id
                        if "—" in product_id:
                            pid_to_use = product_id.split("—")[0].strip()

            # call backend (no top_k kw)
            raw_list = get_top_customers_for_product(pid_to_use)
            if raw_list is None:
                raise RuntimeError(f"No customers returned for product '{pid_to_use}'")
            if not isinstance(raw_list, (list, tuple)):
                try:
                    raw_list = list(raw_list)
                except Exception:
                    raise RuntimeError("Unexpected return type from get_top_customers_for_product")
            cust_ids = list(raw_list)[:20]
            if not cust_ids:
                st.error(f"No customers found for product '{pid_to_use}'")
            else:
                st.session_state.campaign_list = cust_ids
                st.session_state.campaign_index = 0
        except Exception as e:
            st.error(f"Error: {e}")

    if "campaign_list" in st.session_state:
        cust_list = st.session_state.campaign_list
        index = st.session_state.campaign_index

        if cust_list:
            current_id = cust_list[index]
            st.subheader(f"Customer {index+1} of {len(cust_list)} — `{current_id}`")

            # Generate text only for this customer (no images)
            with st.spinner("Generating messages..."):
                try:
                    res = generate_for_customer(
                        current_id,
                        text_model_choice=st.session_state.get("text_model_choice"),
                        custom_model=st.session_state.get("custom_ollama_model"),
                        image_model_choice="No image",
                        creative_kit={"banner": False, "whatsapp": True, "email": True},
                    )
                    if isinstance(res, tuple) and len(res) == 2:
                        data_cam, err_cam = res
                    else:
                        data_cam = res
                        err_cam = None
                except Exception as e:
                    data_cam = None
                    err_cam = str(e)

            if err_cam:
                st.error(err_cam)
            elif data_cam:
                selected = data_cam.get("selected", {})
                st.markdown("### Selected Variant")
                st.markdown(f"> {selected.get('body','')}")

                variants_scored = data_cam.get("variants_scored", [])
                st.markdown("---")
                st.markdown("### A/B/C Variants")
                for item in variants_scored:
                    st.markdown(f"**Variant {item.get('variant', {}).get('variant_tag','?')}** — Score `{item.get('score',0.0)}`")
                    st.markdown(f"> {item.get('variant', {}).get('body','')}")
            else:
                st.error("No data returned for this customer.")

        # Navigation
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅ Previous"):
                st.session_state.campaign_index = max(0, index - 1)
        with col_next:
            if st.button("Next ➡"):
                st.session_state.campaign_index = min(len(cust_list) - 1, index + 1)
