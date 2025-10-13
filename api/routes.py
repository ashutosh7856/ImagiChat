from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from utils.agent_service import process_image_question
from pathlib import Path
import shutil
import uuid

router = APIRouter()

# Directory to temporarily store uploaded images
UPLOAD_DIR = Path(__file__).parent.parent / "uploaded_images"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/home")
async def ask_image_question(
    question: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Endpoint that accepts a single image + question prompt,
    processes it using the AI agent, and returns the response.
    """
    try:
        # Save uploaded image temporarily
        file_ext = Path(image.filename).suffix
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        temp_path = UPLOAD_DIR / temp_filename

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Process via your AI agent
        result = process_image_question(temp_path, question)

        # Delete image after processing (optional but safe)
        temp_path.unlink(missing_ok=True)

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
