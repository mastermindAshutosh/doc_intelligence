# Deployment Guide

To deploy this Document Intelligence app so everyone can use your Streamlit Dashboard, you need to host both the **Backend (FastAPI)** and the **Frontend (Streamlit)** with accessible URLs.

## Option A: Separate Deployments (Recommended)

### 1. Deploy the Backend
Deploy your FastAPI Node to a platform like **Render**, **Railway**, or **AWS** using the `dockerfile`.
- Your public FastAPI URL becomes something like `https://doc-intel-backend.onrender.com`.

### 2. Deploy the Frontend (Streamlit Community Cloud)
1. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and click **New app**.
2. Select your GitHub repository (`mastermindAshutosh/doc_intelligence`).
3. Set the Branch to `master`.
4. Set the Main file path to `frontend/app.py`.
5. **CRITICAL:** Click *Advanced settings...* before deploying to set Secrets!
6. Add your backend URL under **Secrets**:
   ```toml
   API_URL = "https://your-backend-url.onrender.com"
   ```
7. Click **Deploy!**

---

## Option B: Monolithic Deployment (Hugging Face Spaces)

Since you already have a `docker-compose.yml` defining the relative nodes, you can run it as a unified **Docker Space** on Hugging Face.

1. Create a New Space on Hugging Face.
2. Select **Docker** as the SDK template.
3. Push your repository. (You may need to consolidate both frontend and backend into a single `Dockerfile` using ports forwarding or supervisord to run both in one image, as HF spaces run a single container).

Enjoy your live deployment!
