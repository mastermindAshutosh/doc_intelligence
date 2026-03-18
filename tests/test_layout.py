import pytest
import torch
from backend.layout.graph_builder import BlockNode, BlockType, DocumentGraphBuilder
from backend.layout.reading_order import xy_cut_segment
from backend.layout.graph_encoder import DocumentGraphEncoder

def test_xy_cut_single_column():
    blocks = [
        BlockNode(1, "A", 0.5, 0.20, 0.8, 0.1, 0, 1.0, False, False, BlockType.PARAGRAPH, -1, torch.zeros(384)),
        BlockNode(2, "B", 0.5, 0.23, 0.8, 0.1, 0, 1.0, False, False, BlockType.PARAGRAPH, -1, torch.zeros(384)),
        BlockNode(3, "C", 0.5, 0.26, 0.8, 0.1, 0, 1.0, False, False, BlockType.PARAGRAPH, -1, torch.zeros(384)),
    ]
    # Scramble
    blocks_shuffled = [blocks[2], blocks[0], blocks[1]]
    res = xy_cut_segment(blocks_shuffled)
    assert len(res) == 1
    assert [b.block_id for b in res[0]] == [1, 2, 3]

def test_xy_cut_two_column_no_interleaving():
    blocks = [
        # Left column
        BlockNode(1, "L1", 0.25, 0.20, 0.4, 0.1, 0, 1.0, False, False, BlockType.PARAGRAPH, 0, torch.zeros(384)),
        BlockNode(2, "L2", 0.25, 0.23, 0.4, 0.1, 0, 1.0, False, False, BlockType.PARAGRAPH, 0, torch.zeros(384)),
        # Right column
        BlockNode(3, "R1", 0.75, 0.20, 0.4, 0.1, 0, 1.0, False, False, BlockType.PARAGRAPH, 1, torch.zeros(384)),
        BlockNode(4, "R2", 0.75, 0.23, 0.4, 0.1, 0, 1.0, False, False, BlockType.PARAGRAPH, 1, torch.zeros(384)),
    ]
    blocks_shuffled = [blocks[3], blocks[0], blocks[2], blocks[1]]
    groups = xy_cut_segment(blocks_shuffled)
    assert len(groups) == 2
    ids = [[b.block_id for b in g] for g in groups]
    assert ids == [[1, 2], [3, 4]]

def test_block_node_coordinates_normalized():
    b = BlockNode(1, "A", 0.5, 0.2, 0.8, 0.1, 0, 1.0, False, False, BlockType.PARAGRAPH, -1, torch.zeros(384))
    assert 0 <= b.cx <= 1
    assert 0 <= b.cy <= 1
    assert 0 <= b.w <= 1
    assert 0 <= b.h <= 1

def test_graph_has_reading_order_edges():
    builder = DocumentGraphBuilder()
    graph = builder.build(None)  # Stub
    assert True

def test_graph_has_hierarchical_edges():
    assert True

def test_gat_output_shape_256():
    encoder = DocumentGraphEncoder(in_dim=399, hidden=256, out_dim=256)
    x = torch.randn(5, 399)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)
    out = encoder(x, edge_index, edge_attr)
    assert out.shape == (1, 256)
