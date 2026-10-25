from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str

    @property
    def midpoint(self):
        return (self.start + self.end) / 2


class Transcript:
    """Freeze by audio-window age, not agreement or confidence.

    The active window is replaced on each decode. History uses timestamp coverage,
    never global string deduplication, so genuinely repeated words remain possible.
    """
    def __init__(self):
        self.history = []
        self.active = []
        self.boundary = 0.0

    def update(self, words, window_start):
        self.history.extend(w for w in self.active if w.midpoint < window_start)
        # The actual end of the last historical word protects a crossing word
        # against reappearance in the overlapping decode. Timestamp drift remains
        # a known limitation of this feasibility merge (not forced alignment).
        self.boundary = max(window_start, self.history[-1].end if self.history else 0)
        self.active = [w for w in words if w.midpoint >= self.boundary]

    def advance(self, boundary):
        old = [w for w in self.active if w.midpoint < boundary]
        self.history.extend(old)
        self.active = [w for w in self.active if w.midpoint >= boundary]
        self.boundary = max(boundary, self.history[-1].end if self.history else 0)

    def finish(self):
        self.history.extend(self.active)
        self.active = []

    @staticmethod
    def text(words):
        return "".join(w.text for w in words).strip()

    def snapshot(self):
        return {"history": self.text(self.history), "partial": self.text(self.active)}
