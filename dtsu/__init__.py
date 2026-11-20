"""Maneuver-level traffic signal understanding: labels, metrics, traffic rules and RF-DETR decoding.

The task follows Zou et al., "One Perception, All Maneuvers" (arXiv 2609.32316).
"""

DIRECTIONS = ("straight", "left", "right")
COLORS = ("none", "red", "yellow", "green")
PASS = ("stop", "go")


def color_to_pass(color: str) -> str:
    """Signal-implied passability (paper Fig. 2 legend): only red means stop."""
    return "stop" if color == "red" else "go"
