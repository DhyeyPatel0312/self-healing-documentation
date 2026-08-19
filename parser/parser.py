from pydantic import BaseModel,Field

class Info(BaseModel):
    id : str = Field(...,description="Unique Identifier of the chunk, (Module -> Type -> Name)",examples=["module/file_name.py->type->name"])
    type:str = Field(...,description="type of the chunk (function/class/method/variable)")
    name:str = Field(...,description="name of the function/class/method/variable")
    signature:str = Field(...,description="Signature of the function/class/method/variable")
    docstring:str = Field(...,description="Documentation about the chunk")
    code:str = Field(...,description="Actual code of the chunk")
