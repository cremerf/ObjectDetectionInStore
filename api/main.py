'''
Minimal example using fastAPI.
'''
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

# Extra modules
import shutil
import os

# Custom modules
from utils import allowed_file
from utils import get_file_hash
from middleware import model_predict
import settings

# Instantiate API

app = FastAPI()


@app.get("/")
def index():
    return {"Hello": "API"}

@app.post("/")
async def image(image: UploadFile = File(...)):

    # Get image name
    file_name = image.filename

    # No file received, show basic UI
    if not image:
        return {"message": "There is no image"}

    # File received but no filename is provided, show basic UI
    elif file_name == "":
        return {"message": "No image selected for uploading"}


    # File received and it's an image, we must show it and get predictions
    elif image and allowed_file(file_name):
        # In order to correctly display the image in the UI and get model
        # predictions we implement the following:

        # 1. Get an unique file name using utils.get_file_hash() function
        # Create full path to save image
        hashed_name = get_file_hash(image.file, file_name)
        save_path = os.path.join(settings.UPLOAD_FOLDER, hashed_name)

         # 2. Store the image to disk using the new name.
        with open(f"{save_path}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 3. Send the file to be processed by the `model` service            
        prediction, score = model_predict(hashed_name)

        # 4. Update `context` dict with the corresponding values
        context = {
            "prediction": prediction,
            "score": str(score),
            "filename": hashed_name}
    
    return context


@app.post("/hash")
async def image(image: UploadFile = File(...)):

    # Get image name
    file_name = image.filename
    hashed_name = get_file_hash(image.file, file_name)
    return {"hash_name": hashed_name}