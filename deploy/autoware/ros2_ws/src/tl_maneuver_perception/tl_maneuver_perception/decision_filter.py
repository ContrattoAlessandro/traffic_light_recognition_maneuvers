"""Asymmetric temporal filter on the color of the ego's maneuver. No ROS import.

Green needs go_confirm_frames consecutive green frames; red and yellow act at once. A missing light keeps a confirmed
green for at most miss_hold_frames frames and then becomes "none", which Autoware treats as a stop.
With go_confirm_frames=1 and miss_hold_frames=0 the raw per-frame model output is published.
"""
GREEN, NONE = "green", "none"


class DecisionFilter:
    def __init__(self, go_confirm_frames: int, miss_hold_frames: int):
        self.go_confirm, self.miss_hold = max(1, int(go_confirm_frames)), max(0, int(miss_hold_frames))
        self.reset()

    def reset(self) -> None:
        self.output, self.green_run, self.missed = NONE, 0, 0

    def update(self, color: str) -> str:
        if color == GREEN:
            self.green_run += 1
            self.missed = 0
            if self.green_run >= self.go_confirm:
                self.output = GREEN
        elif color == NONE:
            self.green_run = 0
            self.missed += 1
            if self.output != GREEN or self.missed > self.miss_hold:
                self.output = NONE
        else:
            self.green_run, self.missed = 0, 0
            self.output = color
        return self.output
