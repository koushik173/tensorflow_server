from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
import json
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost:3000",  # Replace with your React app's URL
    # Add more origins as needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# MODEL = tf.keras.models.load_model("../Working_CNN/models/1")

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive too "

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    json_data={
        "instances": img_batch.tolist() 
    }
    response = requests.post(endpoint, json=json_data)
    prediction = response.json()["predictions"][0]
    
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = round(100*np.max(prediction), 2)
    response = {
            "class": predicted_class,
            "confidence": confidence
            }
    # json_response = json.dumps(response)
    return {"status": "ok", "data" : response}

if __name__== "__main__":
    uvicorn.run(app, host='localhost', port=8000)
    