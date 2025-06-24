# app/main.py
from fastapi import FastAPI, UploadFile, File
import shutil
from src.predict import predict
import os

app = FastAPI(title="Squid Weight Prediction API")

@app.post("/predict/")
async def predict_squid_weight(file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    weight = predict(temp_path)
    
    # Geçici dosyayı sil
    os.remove(temp_path)
    
    return {"filename": file.filename, "predicted_weight_g": f"{weight:.2f}"}