'''
Minimal example using fastAPI.
'''
from fastapi import FastAPI, UploadFile,File, BackgroundTasks,Request,templating,Form, Cookie, Depends,Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn

# TEST MODULE *--*--*-*-*
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse, Response

# Extra modules
import shutil
import os
from os import getcwd

# Custom modules
from utils import allowed_file
from utils import get_file_hash
from middleware import model_predict
import settings
import shutil
import hashlib
from paths import ODISPaths

PATHS = ODISPaths()

# Instantiate API

app = FastAPI()
#Static file serv
app = FastAPI(title="Service Object Detection")
app.mount("/front",StaticFiles(directory="../api/front"), name="static")


#Jinja2 Template directory
templates = Jinja2Templates(directory="../api/front/templates") #folder templates

PATHS.PREDICTIONS

PATH_FILE = PATHS.PREDICTIONS
PATH_DESTINY = "front/assets/temp/"


@app.get("/", response_class=HTMLResponse)
def root():
    html_address = "../api/front/index.html"

    return FileResponse(html_address, status_code=200)

@app.post("/hash")
async def hash_file(file: UploadFile):
    # Read the file contents into a byte string
    file_bytes = await file.read()
    
    # Calculate the hash of the file contents
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    
    # Return the hash to the client
    return {"file_hash": file_hash}


#@app.post("/", status_code=201)
@app.post("/", status_code=201, response_class=HTMLResponse)
async def image(request:Request, response:Response, image: UploadFile = File(...)):

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

        save_path = os.path.join(PATHS.UPLOAD_FOLDER, hashed_name)

         # 2. Store the image to disk using the new name.
        with open(f"{save_path}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 3. Send the file to be processed by the `model` service            
        status = model_predict(hashed_name)

        # 4. 
        prediction_path = os.path.join(PATHS.PREDICTIONS, hashed_name)
    
        pathpredict_temp = os.path.join(PATH_DESTINY, hashed_name)
        shutil.copy2(prediction_path, pathpredict_temp)

        return templates.TemplateResponse("index2.html",{"request":request,"imgpredict":pathpredict_temp})   
