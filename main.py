import os
import io
import chardet
import google.generativeai as genai
import matplotlib
import pandas as pd
from fastapi import FastAPI,UploadFile,File,Form
from dotenv import load_dotenv
from fastapi.responses import JSONResponse, FileResponse
from fastapi import Form
matplotlib.use("Agg")

load_dotenv(dotenv_path='.env')
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")

app = FastAPI(title="AI Data Analytics Backend")
DATAFRAME = None  



@app.get("/")
def greet():
    return "Hello, World!"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global DATAFRAME
    if file is None:
        return JSONResponse({"error": "No file uploaded"}, status_code=400)

    filename = file.filename
    contents = await file.read()
    try:
        result = chardet.detect(contents[:50000])
        encoding = result["encoding"] or "utf-8"
    
        if filename.endswith(".csv"):
            DATAFRAME = pd.read_csv(io.BytesIO(contents), encoding=encoding)
        elif filename.endswith(".xlsx"):
            DATAFRAME = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse({"error": "Unsupported file format"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"filename": filename, "message": "File uploaded successfully", "rows": DATAFRAME.shape[0], "columns": DATAFRAME.shape[1]}

@app.get("/data")
def profile():
    if DATAFRAME is None:
        return JSONResponse({"error": "No data uploaded yet."}, status_code=400)
    
    df_head = DATAFRAME.head().replace({float("inf"): None, float("-inf"): None}).where(pd.notnull(DATAFRAME.head()), None)

    
    profile = {
        "shape": DATAFRAME.shape,
        "columns": DATAFRAME.dtypes.astype(str).to_dict(),
        "head": df_head.to_dict(orient="records")

    }
    return profile

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
            result = result.replace({float("inf"): None, float("-inf"): None}).where(pd.notnull(result), None).to_dict(orient="records")
        elif isinstance(result, pd.Series):
            result = result.replace({float("inf"): None, float("-inf"): None}).where(pd.notnull(result), None).to_dict()
        else:
            result = result

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return {"query": query, "generated_code": code, "result": result}

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
    
    return FileResponse("plot.png", media_type="image/png")