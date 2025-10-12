from pydantic import BaseModel


class QuestionRequest(BaseModel):
    image_name: str
    question: str


class AgentResponse(BaseModel):
    response: str
    tokens_used: int | None = None
