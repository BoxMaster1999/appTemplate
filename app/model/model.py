from pydantic import BaseModel


class InputBase(BaseModel):
    pass


class InferenceOutput(BaseModel):
    pass


class UploadOutput(BaseModel):
    filename: str
