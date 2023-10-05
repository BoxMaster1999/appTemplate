from fastapi import UploadFile, File, status
from fastapi.routing import APIRouter

from app.apis.v1.model import InputBase, OutputBase
from app.core import ToxisityChacker
from app.config import *

router = APIRouter(prefix="/v1")

toxisity_checker = ToxisityChacker(config.PATH)

@router.post('/toxisity_check',
             description='Проверка текста на токсичность',
             tags=['Inference endpoints'],
             status_code=status.HTTP_200_OK,
             response_model=OutputBase)
def process_base(input_: InputBase) -> OutputBase:
    return OutputBase(text=input_.text, is_toxic=toxisity_checker.check(input_.text))
