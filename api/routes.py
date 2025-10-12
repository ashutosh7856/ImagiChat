from fastapi import APIRouter, HTTPException
from api.schemas import QuestionRequest, AgentResponse
from utils.agent_service import process_image_question

router = APIRouter()


@router.post("/home", response_model=AgentResponse)
async def process_image(request: QuestionRequest):
    try:
        result = process_image_question(request.image_name, request.question)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
