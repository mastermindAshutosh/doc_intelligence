import math
import hashlib

# Simplified dictionary check
ENGLISH_DICT = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"}

def token_entropy(tokens: list[str]) -> float:
    if not tokens: return 0.0
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    ent = 0.0
    for count in freq.values():
        p = count / len(tokens)
        ent -= p * math.log2(p)
    return ent

def text_quality_score(text: str) -> float:
    """
    Returns 0.0 (garbage) to 1.0 (clean). Composite of:
      0.40 * dictionary_word_ratio   (real words / total tokens)
      0.25 * (1 - garbled_char_ratio) (non-ascii / total chars)
      0.20 * min(token_entropy / 4.0, 1.0)
      0.15 * (1 - whitespace_density) (not almost-blank)
    """
    if not text or len(text.strip()) < 20:
        return 0.0
    
    tokens = text.split()
    dict_r  = sum(1 for t in tokens if t.lower() in ENGLISH_DICT or t.lower().isalpha()) / max(len(tokens), 1)
    garbled = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
    entropy = token_entropy(tokens)
    ws_dens = sum(1 for c in text if c in " \t\n") / max(len(text), 1)
    
    return 0.40*dict_r + 0.25*(1-garbled) + 0.20*min(entropy/4.0, 1.0) + 0.15*(1-ws_dens)
