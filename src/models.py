from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Dict, Any


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal['number', 'string', 'boolean']


class FunctionDefinition(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: Dict[str, ParameterDefinition]
    returns: ParameterDefinition


class UserPrompt(BaseModel):
    prompt: str = Field(..., min_length=1)

class FunctionCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    parameters: dict[str, float | int | bool | str]
