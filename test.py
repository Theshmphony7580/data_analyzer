import os
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI app
app = FastAPI(title="AI Data Analyst")

# Global DataFrame
DATAFRAME = None


# ------------------ FILE UPLOAD ------------------
@app.post("/upload")
async def upload_file(file: UploadFile):
    global DATAFRAME
    try:
        if file.filename.endswith(".csv"):
            DATAFRAME = pd.read_csv(file.file)
        elif file.filename.endswith((".xls", ".xlsx")):
            DATAFRAME = pd.read_excel(file.file)
        else:
            return JSONResponse({"error": "Unsupported file format"}, status_code=400)

        return {
            "status": "success",
            "filename": file.filename,
            "columns": list(DATAFRAME.columns),
            "rows": len(DATAFRAME)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ------------------ QUERY HANDLER ------------------
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query(req: QueryRequest):
    global DATAFRAME
    if DATAFRAME is None:
        return JSONResponse({"error": "No data uploaded yet."}, status_code=400)

    prompt = f"""
    You are a data analyst. Convert the following natural language query into valid Pandas code.
    DataFrame is named DATAFRAME.

    Query: "{req.query}"

    Rules:
    - Only return executable Pandas code.
    - Do not redefine DATAFRAME.
    - If aggregation is needed, use groupby(), mean(), sum(), etc.
    - If selecting rows, use loc/iloc.
    - If top values are requested, use head().
    """

    try:
        response = model.generate_content(prompt)
        code = response.text.strip()

        if "```" in code:
            code = code.split("```")[1].replace("python", "").strip()

        result = eval(code, {"DATAFRAME": DATAFRAME, "pd": pd})

        if isinstance(result, pd.DataFrame):
            result = result.head(10).to_dict(orient="records")
        elif isinstance(result, pd.Series):
            result = result.head(10).to_dict()

    except Exception as e:
        return JSONResponse({"error": str(e), "generated_code": code}, status_code=400)

    return {"query": req.query, "generated_code": code, "result": result}


# ------------------ PLOT HANDLER ------------------
class PlotRequest(BaseModel):
    query: str

@app.post("/plot")
def plot(req: PlotRequest):
    global DATAFRAME
    if DATAFRAME is None:
        return JSONResponse({"error": "No data uploaded yet."}, status_code=400)

    prompt = f"""
    You are a data visualization assistant.
    Convert this request into matplotlib code using DATAFRAME.

    Query: "{req.query}"

    Rules:
    - Use plt for plotting.
    - Do not redefine DATAFRAME.
    - Always end with plt.savefig("data/plot.png") instead of plt.show().
    """

    try:
        response = model.generate_content(prompt)
        code = response.text.strip()

        if "```" in code:
            code = code.split("```")[1].replace("python", "").strip()

        os.makedirs("data", exist_ok=True)
        exec(code, {"DATAFRAME": DATAFRAME, "pd": pd, "plt": plt})

    except Exception as e:
        return JSONResponse({"error": str(e), "generated_code": code}, status_code=400)

    return FileResponse("data/plot.png", media_type="image/png")

