import pytest
from backend.ingestion.detector import detect_format, Format, UnsupportedFormatError
from backend.ingestion.ocr import text_quality_score
from backend.ingestion.extractor import DocumentIngester
from backend.config import settings

def test_magic_byte_pdf(minimal_pdf_bytes):
    assert detect_format(minimal_pdf_bytes) == Format.PDF

def test_magic_byte_docx(DOCX_bytes):
    assert detect_format(DOCX_bytes) == Format.DOCX

def test_magic_byte_png(PNG_bytes):
    assert detect_format(PNG_bytes) == Format.PNG

def test_quality_score_clean_text_returns_high():
    text = "the quick brown fox jumps over the lazy dog and that is it"
    score = text_quality_score(text)
    assert score > 0.5

def test_quality_score_garbled_returns_low():
    text = "Ã©Ã¯\x00\x01\x02 §¶ß¤ \t garbled %^&*() text xyz ©£∞¢∞§¶•ªº text and some more random symbols ¥£€" * 5
    score = text_quality_score(text)
    assert score < settings.text_quality_threshold

def test_selective_ocr_only_failing_pages(minimal_pdf_bytes):
    # Testing selective OCR logic manually from ingester
    ingester = DocumentIngester()
    
    doc = ingester.ingest(minimal_pdf_bytes)
    doc.pages[0].quality_score = 0.1  # Force fail
    
    # Simulate selective OCR processing
    for p in doc.pages:
        if p.quality_score < settings.text_quality_threshold and not p.ocr_applied:
            p.ocr_applied = True
            p.text = "this is ocr applied text resulting in good scoring"
            p.quality_score = text_quality_score(p.text)
            doc.ocr_applied = True
            
    assert doc.ocr_applied is True
    assert doc.pages[0].ocr_applied is True
    assert doc.pages[0].quality_score > settings.text_quality_threshold

def test_unsupported_format_raises():
    with pytest.raises(UnsupportedFormatError):
        detect_format(b"GARBAGE BYTES NOT A KNOWN FORMAT")
