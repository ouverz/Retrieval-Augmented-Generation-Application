# app/routers/ingest.py
import os
import shutil
from fastapi import APIRouter, UploadFile, File, Depends
from app.deps import app_container, state_store
from app.schemas.ingest import IngestResponse

router = APIRouter()


@router.post("", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    container=Depends(app_container),
    store=Depends(state_store),
):
    target = os.path.join(container.data_dir, file.filename)
    os.makedirs(container.data_dir, exist_ok=True)
    with open(target, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Invalidate dataset hash so next /init will process
    store.set("dataset_hash", "DIRTY")

    return IngestResponse(
        accepted=True, detail="Uploaded. Call POST /init with force=true to reindex."
    )
