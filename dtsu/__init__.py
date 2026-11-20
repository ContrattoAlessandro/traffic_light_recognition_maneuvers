"""Maneuver-level traffic signal understanding.

A detector reads, for every traffic light in a front-camera image, its pictogram, its color and whether it
belongs to the junction the ego vehicle is approaching. Fixed traffic rules then turn these per-light
attributes into one signal color, and a stop / go decision, per ego maneuver (straight, left, right).

The task and its metrics follow Zou et al., "One Perception, All Maneuvers" (arXiv 2609.32316).
"""

DIRECTIONS = ("straight", "left", "right")
COLORS = ("none", "red", "yellow", "green")
PASS = ("stop", "go")


def color_to_pass(color: str) -> str:
    """Signal-implied passability (Zou et al., Fig. 2): only red means stop."""
    return "stop" if color == "red" else "go"
