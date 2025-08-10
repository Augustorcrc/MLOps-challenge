from pydantic import BaseModel, Field

class VerseRequest(BaseModel):
    reference: str = Field(..., description="Referencia biblica, ej. 'John 3:16'")
   