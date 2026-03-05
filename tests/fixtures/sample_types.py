from pydantic import BaseModel


class CreateInput(BaseModel):
    name: str
    email: str | None = None


class CreateOutput(BaseModel):
    name: str
    email: str | None = None
