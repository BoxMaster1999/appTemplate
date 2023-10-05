from fastapi import UploadFile, File, status
from fastapi.routing import APIRouter

from app.apis.v1.model import InputBase, OutputBase
from app.config import *

from app.core import SafeNaming

router = APIRouter(prefix="/v1")

toxisity_checker = SafeNaming()

@router.post('/toxisity_check',
             description='Проверка текста на токсичность',
             tags=['Inference endpoints'],
             status_code=status.HTTP_200_OK,
             response_model=OutputBase)
def process_base(input_: InputBase) -> OutputBase:
    print(toxisity_checker.check_safety(input_.text))
    return OutputBase(text=input_.text, is_toxic=toxisity_checker.check_safety(input_.text))
