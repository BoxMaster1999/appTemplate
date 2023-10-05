from pydantic import BaseModel


class InputBase(BaseModel):
    pass


class InferenceOutput(BaseModel):
    uuid_: str


class UploadOutput(BaseModel):
    filename: str
