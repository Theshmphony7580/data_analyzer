import streamlit as st
import requests
import pandas as pd
from io import BytesIO


API_URL = "http://127.0.0.1:8000"  # Backend 
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst Dashboard")
st.caption("Upload your dataset and interact using natural language queries and visualizations (powered by Gemini AI + FastAPI).")


uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    with st.spinner("Uploading file..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        resp = requests.post(f"{API_URL}/upload", files=files)
        if resp.status_code == 200:
            st.success(f"File '{uploaded_file.name}' uploaded successfully")
        else:
            st.error(f"Upload failed: {resp.json().get('error', 'Unknown error')}")


if st.button("📄 Show Data Sample"):
    resp = requests.get(f"{API_URL}/data")
    if resp.status_code == 200:
        data = resp.json()
        df = pd.DataFrame(data["head"])
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(resp.json().get("error", "No data uploaded yet."))


st.subheader("🔍 Ask Questions About Your Data")
query = st.text_input("Enter your query (e.g., 'show top 5 customers by revenue')")

if st.button("Run Query"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Analyzing data..."):
            resp = requests.post(f"{API_URL}/query", data={"query": query})
            if resp.status_code == 200:
                result = resp.json()

                res = result.get("result", [])
                if isinstance(res, list):
                    df = pd.DataFrame(res)
                elif isinstance(res, dict):
                    df = pd.DataFrame([res])
                else:
                    df = pd.DataFrame([{"Result": res}])

                st.dataframe(df, use_container_width=True)
            else:
                st.error(f"Error: {resp.text}")


st.subheader("📈 Generate Visualizations")
plot_query = st.text_input("Enter your plot query (e.g., 'plot sales by month')")

if st.button("Generate Plot"):
    if not plot_query:
        st.warning("Please enter a plot query.")
    else:
        with st.spinner("Generating plot..."):
            resp = requests.post(f"{API_URL}/plot", data={"query": plot_query})
            if resp.status_code == 200:
                image_bytes = resp.content
                st.image(image_bytes, caption="Generated Plot", use_column_width=True)
            else:
                st.error(f"Error: {resp.text}")
