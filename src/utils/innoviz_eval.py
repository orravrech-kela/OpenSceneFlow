"""Category taxonomy + label normalisation for Innoviz scene-flow extraction.

Mirrors the AV2 mapping in `src/utils/av2_eval.py` (`CATEGORY_TO_INDEX`) but uses
a compact Innoviz-specific vocabulary derived from the CVAT label sets we have.
Index 0 is reserved for background (no annotation).
"""

from typing import Final, Mapping, Optional

INNOVIZ_CATEGORY_TO_INDEX: Final[Mapping[str, int]] = {
    "NONE": 0,
    "vehicle": 1,
    "person": 2,
    "drone": 3,
    "animal": 4,
}

# CVAT label names are inconsistent across recordings (capitalisation, synonyms).
# Map every encountered raw label to a canonical key in INNOVIZ_CATEGORY_TO_INDEX,
# or None to drop the annotation (e.g. region-of-interest markers like `highway`).
LABEL_NAME_TO_CANONICAL: Final[Mapping[str, Optional[str]]] = {
    "vehicle": "vehicle",
    "car": "vehicle",
    "Vehicle": "vehicle",
    "Car": "vehicle",
    "person": "person",
    "Person": "person",
    "pedestrian": "person",
    "drone": "drone",
    "Drone": "drone",
    "dog": "animal",
    "Dog": "animal",
    "animal": "animal",
    "Animal": "animal",
    # Drop ROI / unused vocab entries:
    "bike": None,
    "Bike": None,
    "highway": None,
    "Highway": None,
}


def canonical_label(raw_name: str) -> Optional[str]:
    """Return the canonical taxonomy key for a CVAT label name, or None to drop."""
    return LABEL_NAME_TO_CANONICAL.get(raw_name, None)


def category_index(canonical_name: Optional[str]) -> int:
    """Return the integer class index for a canonical name; 0 if None/unknown."""
    if canonical_name is None:
        return 0
    return INNOVIZ_CATEGORY_TO_INDEX.get(canonical_name, 0)
