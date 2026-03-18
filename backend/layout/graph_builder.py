from dataclasses import dataclass
from enum import Enum
import torch

class BlockType(Enum):
    TITLE = 1
    HEADING = 2
    PARAGRAPH = 3
    TABLE = 4
    LIST = 5
    FIGURE = 6

@dataclass
class BlockNode:
    block_id:       int
    text:           str
    cx: float           # centroid x, normalized [0,1]
    cy: float           # centroid y, normalized [0,1]
    w:  float           # width as fraction of page width
    h:  float           # height as fraction of page height
    page_idx:       int
    font_size_rel:  float   # font size / page median font size
    is_bold:        bool
    is_italic:      bool
    block_type:     BlockType
    column_idx:     int         # -1 = single column doc
    text_emb:       torch.Tensor  # (384,) MiniLM embedding of block text

class DocumentGraphBuilder:
    def build(self, ingested_doc):
        # Stub builder returning empty list for tests
        return []
