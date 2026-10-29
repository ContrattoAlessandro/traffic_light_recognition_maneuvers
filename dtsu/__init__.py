"""Directional Traffic Signal Understanding (replication of Zou et al.)."""

DIRECTIONS = ("straight", "left", "right", "uturn")
COLORS = ("none", "red", "yellow", "green")
PASS = ("stop", "go")


def color_to_pass(color: str) -> str:
    """Signal-implied passability (paper Fig. 2 legend): only red means stop."""
    return "stop" if color == "red" else "go"
