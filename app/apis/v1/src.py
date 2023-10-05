from fastapi import UploadFile, File, status
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter

import os
import uuid

from app.apis.v1.model import InputBase, InferenceOutput, UploadOutput
from app.config import *

router = APIRouter(prefix="/v1")


@router.post('/processBase',
             description='Process base',
             tags=['Inference endpoints'],
             status_code=status.HTTP_200_OK,
             response_model=InferenceOutput)
def process_base(input_: InputBase) -> InferenceOutput:
    return InferenceOutput()


@router.post('/processImage',
             description='Process image',
             tags=['Inference endpoints'],
             status_code=status.HTTP_200_OK,
             response_model=InferenceOutput)
def process_image(input_image: UploadFile = File(...)) -> InferenceOutput:
    if input_image.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
        raise HTTPException(400, detail='Invalid input type')
    uuid_ = str(uuid.uuid4())
    return InferenceOutput(uuid_=uuid_)


@router.post('/uploadFile',
             description='Upload file',
             tags=['Upload endpoints'],
             status_code=status.HTTP_200_OK,
             response_model=UploadOutput)
def upload_file(input_file: UploadFile = File(...)) -> UploadOutput:
    id_ = uuid.uuid4()
    _, file_extension = os.path.splitext(input_file.filename)
    file_name = f"{id_}.{file_extension}"
    with open(os.path.join(config.STORAGE_PATH, file_name), "wb+") as file_object:
        file_object.write(input_file.file.read())
    return UploadOutput(filename=file_name)
