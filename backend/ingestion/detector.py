from .models import Format
import io
import zipfile

MAGIC_BYTES = {
    b"%PDF":         Format.PDF,
    b"PK\x03\x04":   Format.DOCX,   # ZIP header (DOCX, XLSX, PPTX)
    b"\xff\xd8\xff": Format.JPEG,
    b"\x89PNG":      Format.PNG,
    b"II*\x00":      Format.TIFF,   # little-endian TIFF
    b"MM\x00*":      Format.TIFF,   # big-endian TIFF
}

class UnsupportedFormatError(Exception):
    pass

def _verify_docx_zip(raw: bytes) -> Format:
    # DOCX vs other ZIP: check for word/document.xml inside
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if "word/document.xml" in z.namelist():
                return Format.DOCX
    except zipfile.BadZipFile:
        pass
    raise UnsupportedFormatError(f"Valid PKZIP but not DOCX (no word/document.xml).")

def detect_format(raw: bytes) -> Format:
    for magic, fmt in MAGIC_BYTES.items():
        if raw.startswith(magic):
            # DOCX vs other ZIP: check for word/document.xml inside
            if fmt == Format.DOCX:
                return _verify_docx_zip(raw)
            return fmt
    raise UnsupportedFormatError(f"Unknown format: {raw[:8].hex()}")
