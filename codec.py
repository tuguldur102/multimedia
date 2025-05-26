"""
codec.py   –  Huffman & Shannon-Fano implementations
             (trimmed to run under PyScript)

Dependencies:  collections, heapq, itertools, math, pandas
               (plus typing, ast – all in the standard library)
"""

from __future__ import annotations
import collections, heapq, itertools, math, ast
from typing import List, Tuple
import pandas as pd                      # used only for Shannon-Fano dataframe

# ----------------------------------------------------------------------
# 1. Huffman
# ----------------------------------------------------------------------

class Node:
    """Huffman tree node"""
    _ids = itertools.count()

    def __init__(self, freq, char=None, left=None, right=None):
        self.freq, self.char = freq, char
        self.left, self.right = left, right
        self.id = next(Node._ids)

    def __lt__(self, other):
        return (self.freq, self.id) < (other.freq, other.id)


def build_huffman_tree(text: str) -> Node:
    freqs = collections.Counter(text)
    heap  = [Node(freq, char=c) for c, freq in freqs.items()]
    heapq.heapify(heap)

    # special-case: 1 unique symbol
    if len(heap) == 1:
        only = heapq.heappop(heap)
        heapq.heappush(heap, Node(only.freq, left=only, right=Node(0)))

    while len(heap) > 1:
        n1, n2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, Node(n1.freq + n2.freq, left=n1, right=n2))

    return heap[0]


def gen_codes(node: Node, prefix: str = "", codes=None) -> dict[str, str]:
    if codes is None:
        codes = {}
    if node.char is not None:
        codes[node.char] = prefix or "0"
    else:
        gen_codes(node.left,  prefix + "0", codes)
        gen_codes(node.right, prefix + "1", codes)
    return codes

# ----------------------------------------------------------------------
# 2. Shannon – Fano
# ----------------------------------------------------------------------

class ShannonFano:
    """Shannon-Fano coding"""

    def __init__(self, text: str):
        self.freqs: dict[str, int] = {}
        self.probs: dict[str, float] = {}
        self.codedict: dict[str, str] = {}
        self.fit(text)

    # ------------- public API ----------------
    def encode(self, s: str) -> str:
        return "".join(self.codedict[ch] for ch in s)

    def decode(self, bits: str) -> str:
        rev = {code: sym for sym, code in self.codedict.items()}
        out, buf = [], ""
        for b in bits:
            buf += b
            if buf in rev:
                out.append(rev[buf])
                buf = ""
        return "".join(out)

    def dataframe(self) -> pd.DataFrame:
        rows = []
        for sym, p in sorted(self.probs.items(), key=lambda x: -x[1]):
            rows.append({
                "Symbol": repr(sym),
                "Count":  self.freqs[sym],
                "Probability": round(p, 4),
                "Code":   self.codedict[sym],
                "Code Length": len(self.codedict[sym]),
            })
        return pd.DataFrame(rows)

    # ------------- properties ---------------
    @property
    def entropy(self) -> float:
        return -sum(p * math.log2(p) for p in self.probs.values())

    @property
    def avg_length(self) -> float:
        return sum(self.probs[s] * len(c) for s, c in self.codedict.items())

    # ------------- internal helpers ----------
    def fit(self, text: str):
        self._compute_frequencies(text)
        self._compute_probabilities()
        self._build_codedict()

    def _compute_frequencies(self, text: str):
        self.freqs = dict(collections.Counter(text))

    def _compute_probabilities(self):
        total = sum(self.freqs.values())
        self.probs = {s: c / total for s, c in self.freqs.items()}

    def _build_codedict(self):
        self.codedict = {}
        symbols = sorted(self.probs.items(), key=lambda x: -x[1])
        self._sf_recursive(symbols, "")

    def _sf_recursive(self, symbols: List[Tuple[str, float]], prefix: str):
        if len(symbols) == 1:
            self.codedict[symbols[0][0]] = prefix or "0"
            return
        total = sum(p for _, p in symbols)
        cum, idx = 0.0, 0
        for i, (_, p) in enumerate(symbols):
            cum += p
            idx = i
            if cum >= total / 2:
                break
        left, right = symbols[:idx + 1], symbols[idx + 1:]
        self._sf_recursive(left,  prefix + "0")
        self._sf_recursive(right, prefix + "1")

# ----------------------------------------------------------------------
# 3. Convenience wrapper used by index.html
# ----------------------------------------------------------------------

def encode(text: str, algorithm: str = "Huffman"):
    """
    Helper so the front-end can call one function and get:
      root, codes, stats-rows
    """
    if algorithm == "Huffman":
        root = build_huffman_tree(text)
        codes = gen_codes(root)
    else:
        sf = ShannonFano(text)
        root = None           # not used by the HTML for this path
        codes = sf.codedict
    freqs = collections.Counter(text)
    total = len(text)
    rows = [(c, f, f / total, codes[c]) for c, f in freqs.items()]
    return root, codes, rows
