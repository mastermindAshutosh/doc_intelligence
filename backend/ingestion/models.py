from pydantic import BaseModel
from enum import Enum

class Format(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PNG = "png"
    TIFF = "tiff"
    JPEG = "jpeg"

class PageData(BaseModel):
    page_idx:     int
    text:         str
    quality_score: float        # 0.0 – 1.0
    ocr_applied:  bool
    width_pt:     float         # points (PDF native)
    height_pt:    float
    dpi:          int | None    # None for digital PDFs

class IngestedDoc(BaseModel):
    doc_hash:     str            # SHA256 of raw bytes
    format:       Format
    pages:        list[PageData]
    page_count:   int
    ocr_applied:  bool           # any page used OCR
    has_tables:   bool           # detected by pdfplumber
    has_images:   bool
    metadata:     dict           # author, created_date, title if available
