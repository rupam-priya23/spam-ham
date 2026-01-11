import streamlit as st
from spam_classifier import train_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #f9fafb;
}

/* Title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #1f2937;
    text-align: center;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Card */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.06);
}

/* Buttons */
.spam-btn > button {
    background-color: #dc2626;
    color: white;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 600;
}

.safe-btn > button {
    background-color: #16a34a;
    color: white;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return train_model()

model, vectorizer, clean_text = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 📌 Project Info")
st.sidebar.write(
    """
    **Spam Message Classifier**

    This ML-powered app detects whether a message
    is **Spam** or **Not Spam**.

    ### 🛠 Tech Stack
    - Python
    - TF-IDF
    - Naive Bayes
    - Streamlit
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 **Built by You**")

# ---------------- MAIN UI ----------------
st.markdown("<div class='main-title'>📨 Spam Message Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Check whether a message is spam or safe</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### ✍️ Enter your message")
user_input = st.text_area(
    "",
    placeholder="Type your SMS or email message here...",
    height=160
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='safe-btn'>", unsafe_allow_html=True)
    check_safe = st.button("✅ Check Message", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='spam-btn'>", unsafe_allow_html=True)
    clear = st.button("🗑️ Clear Text", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- LOGIC ----------------
if clear:
    st.experimental_rerun()

if check_safe:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message before checking.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        st.markdown("---")

        if prediction == 1:
            st.error("🚨 **Spam Detected** — This message looks suspicious.")
        else:
            st.success("✅ **Safe Message** — This message looks genuine.")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <br><hr>
    <p style="text-align:center; color:#6b7280; font-size:14px;">
    Built with ❤️ using Machine Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
