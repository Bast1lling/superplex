"""Hand-written plausible SQ counts per class (my judgment of what a person would draw), and the share of
instances whose K lies in that range. Classes not in the table are ignored. Walls: a Replica wall instance can
span several faces with door / window openings, hence up to 4."""

RANGES = {
    "floor": (1, 1), "ceiling": (1, 2), "rug": (1, 1), "wall": (1, 4), "blinds": (1, 1), "picture": (1, 1),
    "door": (1, 1), "window": (1, 2), "cushion": (1, 1), "pillow": (1, 1), "book": (1, 1), "vent": (1, 1),
    "switch": (1, 1), "wall-plug": (1, 1), "plate": (1, 1), "tv-screen": (1, 1), "clock": (1, 1), "panel": (1, 1),
    "tablet": (1, 1), "tissue-paper": (1, 1), "camera": (1, 1), "candle": (1, 1), "stool": (1, 2), "cabinet": (1, 2),
    "nightstand": (1, 2), "bin": (1, 1), "basket": (1, 1), "vase": (1, 2), "pot": (1, 1), "blanket": (1, 2),
    "comforter": (1, 2), "chair": (1, 3), "sofa": (1, 4), "bed": (1, 3), "table": (1, 5), "lamp": (1, 3),
    "pillar": (1, 1), "indoor-plant": (1, 2), "plant-stand": (1, 2), "desk-organizer": (1, 1),
}


def plausibility(per_instance: dict) -> tuple[float, float, float]:
    """-> (share in range, share over-split, share under-split / empty)."""
    n = inside = over = under = 0
    for name, k in zip(per_instance["name"], per_instance["K"]):
        if name not in RANGES:
            continue
        lo, hi = RANGES[name]
        n += 1
        inside += lo <= k <= hi
        over += k > hi
        under += k < lo
    return inside / max(n, 1), over / max(n, 1), under / max(n, 1)
