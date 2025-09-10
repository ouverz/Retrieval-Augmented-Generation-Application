# app/routers/ingest.py
import os
import shutil
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from backend.dependencies import app_container, state_store
from backend.schemas.ingest import IngestResponse
from backend.security import require_api_key

logger = logging.getLogger(__name__)

# File upload security configuration
ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_FILENAME_LENGTH = 255

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def validate_file_upload(file: UploadFile) -> None:
    """Validate file upload for security compliance.
    
    Args:
        file: The uploaded file to validate
        
    Raises:
        HTTPException: If file fails validation
    """
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    # Validate filename
    if not file.filename:
        raise HTTPException(400, "Filename is required")
    
    if len(file.filename) > MAX_FILENAME_LENGTH:
        raise HTTPException(400, f"Filename too long. Maximum length: {MAX_FILENAME_LENGTH}")
    
    # Check for path traversal attacks
    if ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(400, "Invalid filename. Path traversal detected")
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise HTTPException(400, f"File type not allowed. Allowed types: {allowed}")
    
    # Validate MIME type
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(400, "Invalid content type. Only PDF files are allowed")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove any path components
    filename = os.path.basename(filename)
    
    # Replace any potentially dangerous characters
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Ensure filename doesn't start with dot (hidden file)
    if filename.startswith('.'):
        filename = 'file_' + filename
    
    return filename


@router.post("", response_model=IngestResponse)
@limiter.limit("5/hour")
async def ingest_pdf(
    request: Request,
    file: UploadFile = File(...),
    container=Depends(app_container),
    store=Depends(state_store),
    api_key: str = Depends(require_api_key)
):
    """Upload and ingest a PDF file with security validation.
    
    Args:
        file: PDF file to upload
        container: Application container
        store: State store
        
    Returns:
        IngestResponse with upload status
        
    Raises:
        HTTPException: If file validation fails
    """
    try:
        # Validate file upload
        validate_file_upload(file)
        
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        
        # Create secure target path
        data_dir = Path(container.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        target_path = data_dir / safe_filename
        
        # Check if file already exists and create unique name if needed
        counter = 1
        original_target = target_path
        while target_path.exists():
            stem = original_target.stem
            suffix = original_target.suffix
            target_path = data_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # Save file securely
        with open(target_path, "wb") as out:
            # Read file in chunks to prevent memory issues
            chunk_size = 8192
            total_size = 0
            
            while chunk := await file.read(chunk_size):
                total_size += len(chunk)
                
                # Double-check file size during upload
                if total_size > MAX_FILE_SIZE:
                    # Clean up partial file
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(400, "File too large")
                
                out.write(chunk)
        
        logger.info(f"File uploaded successfully: {safe_filename} ({total_size} bytes)")
        
        # Invalidate dataset hash so next /init will process
        store.set("dataset_hash", "DIRTY")
        
        return IngestResponse(
            accepted=True, 
            detail=f"Uploaded '{safe_filename}'. Call POST /init with force=true to reindex."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(500, "Upload failed")
