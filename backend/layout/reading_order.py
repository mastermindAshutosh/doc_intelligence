from .graph_builder import BlockNode

def _largest_vertical_gap(blocks: list[BlockNode]) -> tuple[float, float]:
    if len(blocks) < 2: return 0.0, 0.0
    s = sorted(blocks, key=lambda b: b.cx)
    max_gap, pos = 0.0, 0.0
    for i in range(len(s)-1):
        g = s[i+1].cx - (s[i].cx + s[i].w/2)
        if g > max_gap:
            max_gap, pos = g, s[i].cx + s[i].w/2 + g/2
    return max_gap, pos

def _largest_horizontal_gap(blocks: list[BlockNode]) -> tuple[float, float]:
    if len(blocks) < 2: return 0.0, 0.0
    s = sorted(blocks, key=lambda b: b.cy)
    max_gap, pos = 0.0, 0.0
    for i in range(len(s)-1):
        g = s[i+1].cy - (s[i].cy + s[i].h/2)
        if g > max_gap:
            max_gap, pos = g, s[i].cy + s[i].h/2 + g/2
    return max_gap, pos

def xy_cut_segment(blocks: list[BlockNode]) -> list[list[BlockNode]]:
    if len(blocks) <= 1:
        return [blocks]
    v_gap, v_pos = _largest_vertical_gap(blocks)
    if v_gap > 0.05:
        left  = [b for b in blocks if (b.cx + b.w/2) < v_pos]
        right = [b for b in blocks if (b.cx + b.w/2) >= v_pos]
        return xy_cut_segment(left) + xy_cut_segment(right)
    h_gap, h_pos = _largest_horizontal_gap(blocks)
    if h_gap > 0.04:
        top    = [b for b in blocks if (b.cy + b.h/2) < h_pos]
        bottom = [b for b in blocks if (b.cy + b.h/2) >= h_pos]
        return xy_cut_segment(top) + xy_cut_segment(bottom)
    return [sorted(blocks, key=lambda b: (b.page_idx, b.cy))]
