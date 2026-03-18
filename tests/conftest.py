import pytest
import torch

@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    """Minimal valid PDF with text layer. No OCR needed."""
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

@pytest.fixture
def scanned_pdf_bytes() -> bytes:
    """PDF with no text layer — should trigger OCR."""
    return b"%PDF-1.4\n2 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

@pytest.fixture
def DOCX_bytes() -> bytes:
    import io, zipfile
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        z.writestr("word/document.xml", "<w:document></w:document>")
    return out.getvalue()

@pytest.fixture
def PNG_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n"

@pytest.fixture
def in_dist_logits() -> torch.Tensor:
    """Logits from a confident in-distribution prediction."""
    return torch.tensor([[8.5, 0.1, 0.2, 0.1, 0.1]])  # TAX

@pytest.fixture
def ood_logits() -> torch.Tensor:
    """Logits from an OOD document — roughly uniform."""
    return torch.tensor([[0.3, 0.2, 0.3, 0.1, 0.1]])
