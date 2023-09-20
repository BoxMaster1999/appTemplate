from fastapi import FastAPI, UploadFile, File, status
from fastapi.exceptions import HTTPException

import os
import uuid

from .model import *
from .config import *

app = FastAPI(title='appTemplate',
              description='Fastapi template',
              version='0.1')


@app.get('/health',
         tags=['System probs'])
def health() -> int:
    return status.HTTP_200_OK


@app.post('/processBase',
          description='Process base',
          tags=['Inference endpoints'],
          status_code=status.HTTP_200_OK,
          response_model=model.InferenceOutput)
def process_base(input_: model.InputBase) -> model.InferenceOutput:
    return model.InferenceOutput()


@app.post('/processImage',
          description='Process image',
          tags=['Inference endpoints'],
          status_code=status.HTTP_200_OK,
          response_model=model.InferenceOutput)
def process_image(input_image: UploadFile = File(...)) -> model.InferenceOutput:
    if input_image.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
        raise HTTPException(400, detail='Invalid input type')
    return model.InferenceOutput()


@app.post('/uploadFile',
          description='Upload file',
          tags=['Upload endpoints'],
          status_code=status.HTTP_200_OK,
          response_model=model.UploadOutput)
def upload_file(input_file: UploadFile = File(...)) -> model.UploadOutput:
    id_ = uuid.uuid4()
    _, file_extension = os.path.splitext(input_file.filename)
    file_name = f"{id_}.{file_extension}"
    with open(os.path.join(config.STORAGE_PATH, file_name), "wb+") as file_object:
        file_object.write(input_file.file.read())
    return model.UploadOutput(filename=file_name)
