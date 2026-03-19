import streamlit as st
import requests
import json
from datetime import datetime

# Configure API URL (Use Streamlit Secrets for deployments, fallback to local docker address)
API_URL = st.secrets.get("API_URL", "https://doc-intelligence-backend-bl5l.onrender.com")

st.set_page_config(
    page_title="Document Intelligence Dashboard",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Intelligence System")
st.markdown("---")

# Sidebar - Monitoring Metrics
st.sidebar.title("System Health & Metrics")
st.sidebar.markdown("[🔗 Backend API Documentation](http://localhost:8000/docs)")
st.sidebar.markdown("---")

try:
    mon_res = requests.get(f"{API_URL}/monitoring")
    if mon_res.status_code == 200:
        metrics = mon_res.json()
        st.sidebar.metric("Rolling ECE", f"{metrics.get('ece_rolling_7d', 0.0):.4f}")
        st.sidebar.metric("Uncertain Rate (24h)", f"{metrics.get('uncertain_rate_24h', 0.0)*100:.1f}%")
        st.sidebar.metric("OCR Quality", f"{metrics.get('ocr_quality_p10', 0.0)*100:.1f}%")
        
        drift = metrics.get("drift_flags", {})
        if drift.get("overall", False):
            st.sidebar.error("🚨 CONFIDENCE DRIFT DETECTED")
        else:
            st.sidebar.success("✅ No Confidence Drift")
    else:
        st.sidebar.warning("Unable to fetch real-time monitoring stats.")
except Exception:
    st.sidebar.warning("Backend API might be offline.")

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
        
        if st.button("Classify Document", type="primary"):
            with st.spinner("Processing metadata, layout, and text..."):
                try:
                    res = requests.post(f"{API_URL}/classify", files=files)
                    if res.status_code == 200:
                        st.session_state["last_result"] = res.json()
                        st.success("Classification Completed!")
                    else:
                        st.error(f"Error classifying document: {res.text}")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")

with col2:
    st.header("Results & Routing")
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Decision")
            st.markdown(f"## **`{res['prediction']}`**")
        
        with c2:
            st.subheader("Confidence")
            st.markdown(f"## **`{res['confidence']*100:.2f}%`**")
            
        st.markdown("---")
        st.subheader("Routing Decision")
        routing = res["routing"].upper()
        if routing == "DIRECT":
            st.success(f"➡️ **{routing}** (Auto-Approved)")
        elif routing == "ASYNC_CONFIRM":
            st.warning(f"➡️ **{routing}** (Requires Minor Review)")
        elif routing == "HUMAN_REVIEW":
            st.error(f"➡️ **{routing}** (Requires Full Audit)")
        else: # OOD
            st.error(f"➡️ **{routing}** (Out of Distribution detected!)")
            
        with st.expander("Show Metadata & Details"):
            st.json(res)
    else:
        st.info("Upload a document and press Classify to see insights.")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
