from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import gradio as gr
import uvicorn
import joblib
import base64
import io
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ✅ Load the models
clv_model = joblib.load("clv_model.pkl")
churn_model = joblib.load("churn_model.pkl")

# Prediction function for Gradio
def predict_clv_churn(file):
    df = pd.read_excel(file)
    features = df[['age', 'income', 'tenure', 'num_purchases']]
    df['Predicted_CLV'] = clv_model.predict(features)
    df['Churn_Probability'] = churn_model.predict_proba(features)[:, 1]
    return df

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/demo", response_class=HTMLResponse)
async def demo(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/technical", response_class=HTMLResponse)
async def technical(request: Request):
    return templates.TemplateResponse("technical.html", {"request": request})

# ✅ Fix for main prediction route
@app.post("/predict-clv-churn")
async def predict_clv_churn_api(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith((".xlsx", ".xls")):
            error_msg = "Only Excel files (.xlsx, .xls) are supported."
            logger.error(f"File type validation failed: {error_msg}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": error_msg}
            )

        contents = await file.read()
        if not contents:
            error_msg = "Uploaded file is empty."
            logger.error(error_msg)
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": error_msg}
            )

        logger.info(f"File size: {len(contents)} bytes")
        
        # Read Excel file
        df = pd.read_excel(io.BytesIO(contents))
        logger.info(f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")

        required_cols = ['age', 'income', 'tenure', 'num_purchases']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            error_msg = f"Missing required columns: {', '.join(missing)}. Required columns are: {', '.join(required_cols)}"
            logger.error(error_msg)
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": error_msg}
            )

        features = df[required_cols]
        logger.info("Making predictions...")

        # Make predictions
        df['Predicted_CLV'] = clv_model.predict(features)
        df['Churn_Probability'] = churn_model.predict_proba(features)[:, 1]

        # Return results (limit to first 10 rows for display)
        results = df[['Predicted_CLV', 'Churn_Probability']].head(10).to_dict(orient="records")
        
        logger.info(f"Predictions completed successfully for {len(results)} rows")
        return JSONResponse({"success": True, "results": results})

    except pd.errors.EmptyDataError:
        error_msg = "The Excel file appears to be empty or corrupted."
        logger.error(error_msg)
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": error_msg}
        )
    except pd.errors.ParserError:
        error_msg = "Unable to parse the Excel file. Please ensure it's a valid Excel format."
        logger.error(error_msg)
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": error_msg}
        )
    except Exception as e:
        error_msg = f"Error processing the Excel file: {str(e)}"
        logger.error(f"❌ Error in /predict-clv-churn: {error_msg}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

# ✅ For full Excel download
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))

        features = df[['age', 'income', 'tenure', 'num_purchases']]
        df['Predicted_CLV'] = clv_model.predict(features)
        df['Churn_Probability'] = churn_model.predict_proba(features)[:, 1]

        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        encoded = base64.b64encode(output.read()).decode()

        return JSONResponse({
            "success": True,
            "message": "Prediction successful.",
            "excel_base64": encoded
        })
    except Exception as e:
        logger.error(f"❌ Error in /predict/: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Error processing the Excel file. Please check your file and try again."}
        )

# ✅ Gradio app
gradio_interface = gr.Interface(
    fn=predict_clv_churn,
    inputs=gr.File(label="Upload Excel File"),
    outputs=gr.Dataframe(headers=["Predicted_CLV", "Churn_Probability"], type="pandas"),
    title="CLV & Churn Predictor",
    description="Upload a valid Excel file with customer data to get CLV and Churn predictions."
)

# ✅ Mount Gradio
app = gr.mount_gradio_app(app, gradio_interface, path="/gradio")

# ✅ Run locally
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)