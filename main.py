import os
import io
import google.generativeai as genai
import matplotlib
import gradio as gr
import pandas as pd
import chardet
from fastapi import FastAPI,UploadFile,File,Form
from dotenv import load_dotenv
from fastapi.responses import JSONResponse, FileResponse
from fastapi import Form
matplotlib.use("Agg")   # ✅ Fix backend issue
import matplotlib.pyplot as plt
# from fastapi.responses import FileResponse

#API key
load_dotenv(dotenv_path='.env')
#intiates the api request to Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-lite-001")
app = FastAPI(title="AI Data Analytics Backend")
DATAFRAME = None  



@app.get("/")
def greet():
    return "Hello, World!"

#user uploads file
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global DATAFRAME
    if file is None:
        return JSONResponse({"error": "No file uploaded"}, status_code=400)

    filename = file.filename
    contents = await file.read()
    try:
        if filename.endswith(".csv"):
            DATAFRAME = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx"):
            DATAFRAME = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse({"error": "Unsupported file format"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"filename": filename, "message": "File uploaded successfully", "rows": DATAFRAME.shape[0], "columns": DATAFRAME.shape[1]}

# user requests data profile
@app.get("/data")
def profile():
    if DATAFRAME is None:
        return JSONResponse({"error": "No data uploaded yet."}, status_code=400)
    
    profile = {
        "shape": DATAFRAME.shape,
        "columns": DATAFRAME.dtypes.astype(str).to_dict(),
        "head": DATAFRAME.head().to_dict(orient="records")

    }
    return profile

# user queries data
@app.post("/query")
def query(query: str = Form(...)):
    global DATAFRAME
    if DATAFRAME is None:
        return JSONResponse({"error": "No data uploaded yet."}, status_code=400)

    prompt = f"""You are a data analyst.  
        Convert the following natural language query into Python Pandas code.  
        The DataFrame is named DATAFRAME.  
        Query: "{query}"  

        Rules:
        - Output only the Pandas code, nothing else.
        - Do not include explanations or text.
        - Do not redefine DATAFRAME.
        - If the query involves aggregation, use Pandas methods (groupby, mean, sum, etc.).
        - If the query asks for top rows, use head().
        - Always return a DataFrame or Series as the final object.
        """

    try:
        response = model.generate_content(prompt)
        code = response.text.strip()

        if "```" in code:
            code = code.split("```")[1].replace("python", "").strip()

        result = eval(code)
        if isinstance(result, pd.DataFrame):
            result = result.to_dict(orient="records")
        elif isinstance(result, pd.Series):
            result = result.to_dict()
        else:
            result = result

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return {"query": query, "generated_code": code, "result": result}

#plots the data
@app.post("/plot")
def plot(query: str = Form(...)):
    global DATAFRAME
    if DATAFRAME is None:
        return JSONResponse({"error": "No data uploaded yet."}, status_code=400)

    prompt = f"""You are a data visualization assistant.  
        Convert the following natural language query into Python code using Pandas and matplotlib.  
        The DataFrame is named DATAFRAME.  

        Query: "{query}"  

        Rules:
        - Output only valid Python code, nothing else.
        - Always use matplotlib.pyplot as plt.
        - Always finish with plt.savefig("plot.png") instead of plt.show().
        - Do not redefine DATAFRAME.
        - Keep plots simple (line, bar, scatter, histogram).
        """

    try:
        response = model.generate_content(prompt)
        code = response.text.strip()

        if "```" in code:
            code = code.split("```")[1].replace("python", "").strip()

        exec(code, globals())

    except Exception as e:
        return JSONResponse(
            {"error": str(e), "generated_code": code if 'code' in locals() else None},
            status_code=400,
        )
def gradio_upload(file):
    global DATAFRAME
    if file is None:
        return "❌ No file uploaded"
    filename = file.name if hasattr(file, "name") else str(file)
    try:
        if filename.endswith(".csv"):
            with open(filename, 'rb') as f:
                raw = f.read(50000)  # Read first 50,000 bytes
                result = chardet.detect(raw)
                encoding = result['encoding'] or 'utf-8'
                f.seek(0)
                DATAFRAME = pd.read_csv(f, encoding=encoding)
        elif filename.endswith(".xlsx"):
            with open(filename, 'rb') as f:
                DATAFRAME = pd.read_excel(f)
        else:
            return "❌ Unsupported file format"
    except Exception as e:
        return f"❌ Error: {str(e)}"
    return f"✅ File '{filename}' uploaded successfully. Rows: {DATAFRAME.shape[0]}, Columns: {DATAFRAME.shape[1]}"

def gradio_query(query):
    global DATAFRAME
    if DATAFRAME is None:
        return []
    prompt = f"""You are a data analyst.  
        Convert the following natural language query into Python Pandas code.  
        The DataFrame is named DATAFRAME.  
        Query: "{query}"  

        Rules:
        - Output only the Pandas code, nothing else.
        - Do not include explanations or text.
        - Do not redefine DATAFRAME.
        - If the query involves aggregation, use Pandas methods (groupby, mean, sum, etc.).
        - If the query asks for top rows, use head().
        - Always return a DataFrame or Series as the final object.
        """
    try:
        response = model.generate_content(prompt)
        code = response.text.strip()
        if "```" in code:
            code = code.split("```")[1].replace("python", "").strip()
        result = eval(code)
        if isinstance(result, pd.DataFrame):
            return result
        elif isinstance(result, pd.Series):
            return pd.DataFrame(result)
        else:
            return pd.DataFrame([{"Result": result}])
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])

def gradio_plot(query):
    global DATAFRAME
    if DATAFRAME is None:
        return None
    prompt = f"""You are a data visualization assistant.  
        Convert the following natural language query into Python code using Pandas and matplotlib.  
        The DataFrame is named DATAFRAME.  

        Query: "{query}"  

        Rules:
        - Output only valid Python code, nothing else.
        - Always use matplotlib.pyplot as plt.
        - Always finish with plt.savefig("plot.png") instead of plt.show().
        - Do not redefine DATAFRAME.
        - Keep plots simple (line, bar, scatter, histogram).
        """
    try:
        response = model.generate_content(prompt)
        code = response.text.strip()
        if "```" in code:
            code = code.split("```")[1].replace("python", "").strip()
        exec(code, globals())
        return "plot.png"
    except Exception as e:
        return None

with gr.Blocks() as demo:
    gr.Markdown("# 📊 AI Data Analyst Prototype ")

    with gr.Row():
        file_input = gr.File(label="Upload CSV/Excel", file_types=[".csv", ".xlsx"])
        upload_output = gr.Textbox(label="Upload Status")

    file_input.change(gradio_upload, inputs=file_input, outputs=upload_output)

    with gr.Row():
        plot_input = gr.Textbox(label="Ask for a plot")
        plot_output = gr.Image(label="Generated Plot", type="filepath")
        plot_button = gr.Button("Generate Plot")

    plot_button.click(fn=gradio_plot, inputs=plot_input, outputs=plot_output)

    # Show the plot after it's generated
    # def show_plot(plot_path):
    #     if plot_path and os.path.exists(plot_path):
    #         return plot_path
    #     return None

    # plot_button.click(gradio_plot, inputs=plot_input, outputs=plot_output).then(
    #     show_plot, inputs=plot_output, outputs=plot_output
    # )

    with gr.Row():
        query_input = gr.Textbox(label="Ask a question about your data")
        query_output = gr.Dataframe(label="Query Result")
        query_button = gr.Button("Run Query")
    query_button.click(gradio_query, inputs=query_input, outputs=query_output)

demo.launch()