from pydantic import BaseModel


class InputBase(BaseModel):
    text: str


class OutputBase(BaseModel):
    text: str
    is_toxic: bool
