import base64
import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import plotly.express as px



API_URL = "http://127.0.0.1:8000"  # Backend 
st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("AI Data Analyst 📊")

# Create clean tabs
tab1, tab2, tab3 = st.tabs(["Upload & Preview", "Query Data", "Visualization"])


uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

with tab1:
    if uploaded_file is not None:
        with st.spinner("Uploading file..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            resp = requests.post(f"{API_URL}/upload", files=files)
            if resp.status_code == 200:
                st.success(f"File '{uploaded_file.name}' uploaded successfully")
            else:
                st.error(f"Upload failed: {resp.json().get('error', 'Unknown error')}")


if st.button("Show Data Sample"):
    resp = requests.get(f"{API_URL}/data")
    if resp.status_code == 200:
        data = resp.json()
        df = pd.DataFrame(data["head"])
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(resp.json().get("error", "No data uploaded yet."))

with tab2:
    st.subheader("Ask Questions About Your Data")
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


with tab3:
    st.subheader("Generate Visualizations")

    st.markdown("### AI-Powered Plot Generation")
    plot_query = st.text_input("Enter your plot query (e.g., 'plot sales by month')")

    if st.button("Generate Plot"):
        if not plot_query:
            st.warning("Please enter a plot query.")
        else:
            with st.spinner("Generating plot..."):
                resp = requests.post(f"{API_URL}/plot", data={"query": plot_query})
                if resp.status_code == 200:
                    image_data = base64.b64decode(resp.json().get("image_base64"))
                    st.image(image_data, caption="Generated Plot", use_container_width=True)
                else:
                    st.error(f"Error: {resp.text}")

    st.markdown("---")

    # st.markdown("###  Create Interactive Plot (Manual)")

    if 'df_preview' in st.session_state and st.session_state.df_preview is not None:
        try:
            import plotly.express as px
        except Exception:
            px = None
            st.error("The 'plotly' package is not installed or couldn't be imported. Install it with 'pip install plotly' and restart the app.")
        df = st.session_state.df_preview.copy()

        col_x = st.selectbox("Select X-axis", options=df.columns)
        col_y = st.selectbox("Select Y-axis", options=df.columns)
        chart_type = st.selectbox("Chart type", ["Line", "Bar", "Scatter", "Histogram", "Box"])

        if st.button("📊 Create Interactive Plot"):
            if px is None:
                st.warning("Cannot create interactive plot because 'plotly' is not available.")
            else:
                try:
                    if chart_type == "Line":
                        fig = px.line(df, x=col_x, y=col_y)
                    elif chart_type == "Bar":
                        fig = px.bar(df, x=col_x, y=col_y)
                    elif chart_type == "Scatter":
                        fig = px.scatter(df, x=col_x, y=col_y)
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=col_x)
                    elif chart_type == "Box":
                        fig = px.box(df, x=col_x, y=col_y)
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Plot error: {e}")
    else:
        st.info("Please load a dataset in the 'Upload & Preview' tab first.")

