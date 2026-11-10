from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from uuid import uuid4

from backend.services.image_service import validate_image


router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):

    # Check whether a file was uploaded
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded."
        )

    # Generate a unique filename
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    filename = f"{uuid4()}{file_extension}"

    file_path = UPLOAD_DIR / filename

    # Save the uploaded image
    contents = await file.read()

    file_path.write_bytes(contents)

    # Validate the image
    result = validate_image(file_path)

    if not result["valid"]:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=result["message"]
        )

    return {
        "message": "Image uploaded successfully.",
        "filename": filename,
        "validation": result["message"]
    }