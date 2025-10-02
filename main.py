from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from PIL import Image
import pytesseract
import io
import re
from openai import OpenAI
import os

STATIC_PROMPT="You are an assistant that cleans OCR text and converts it into clear, meaningful questions with four multiple-choice options."

# Set your OpenAI API key here or better via environment variable
app = FastAPI()

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def clean_ocr_text(text: str) -> str:
    """
    Post-process OCR text:
    - Remove empty lines and artifacts
    - Normalize spaces
    - Keep numbers, percentages, and words intact
    """
    # Remove stray characters and pipes
    text = re.sub(r"[|_]", " ", text)
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Keep only meaningful lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

@app.post("/")
# async def extract_text_from_image(files: list[UploadFile] = File(...)):
async def process_inputs(
    files: Optional[List[UploadFile]] = File(None),
    user_text: Optional[str] = Form(None)            # Optional user text
):
    results = []
    # Handle images if provided
    if files:
        for file in files:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
            # Use Tesseract with better page segmentation
            extracted_text = pytesseract.image_to_string(img, config="--psm 6")
            cleaned_text = clean_ocr_text(extracted_text)

            # Combine static prompt + image text + user_text
            combined_text = f"{STATIC_PROMPT}\n\nImage Text:\n{cleaned_text}\n\nUser Text:\n{user_text or ''}"

            # Call OpenAI Chat API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": combined_text}
                ]
            )

            openai_response = response.choices[0].message.content

            results.append({
                "ocr_raw": extracted_text,
                "ocr_cleaned": cleaned_text,
                "openai_response": openai_response
            })

    # If no images but user_text is provided
    if not files and user_text:
        combined_text = f"{STATIC_PROMPT}\n\nUser Text:\n{user_text}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": combined_text}
            ]
        )

        openai_response = response.choices[0].message.content
        results.append({
            "combined_text": combined_text,
            "openai_response": openai_response
        })

    # If neither images nor text provided, return empty list or message
    if not files and not user_text:
        return JSONResponse(content={"message": "No input provided", "results": []})

    return JSONResponse(content={"results": results})