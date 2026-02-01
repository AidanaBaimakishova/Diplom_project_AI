from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse

from pipeline import process_file
from config import UPLOAD_DIR

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def healthcheck():
    return {"status": "ok", "service": "blood-analysis-ocr"}


@app.post("/recognize", response_class=PlainTextResponse)
async def recognize(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result_text = process_file(save_path)
    return result_text
