from fastapi import UploadFile, File, status
from fastapi.routing import APIRouter

from app.apis.v1.model import InputBase, OutputBase
from app.config import *

from app.core import safe_naming_checker

router = APIRouter(prefix="/v1")


print(safe_naming_checker.check_safety('Привет'))

@router.post('/toxisity_check',
             description='Проверка текста на токсичность',
             tags=['Inference endpoints'],
             status_code=status.HTTP_200_OK,
             response_model=OutputBase)
def process_base(input_: InputBase) -> OutputBase:
    return OutputBase(text=input_.text, is_toxic=safe_naming_checker.check_safety(input_.text))
