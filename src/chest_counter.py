"""Manual per-run chest counts, independent of map and projection sessions."""
from __future__ import annotations


CHEST_ACTIONS = {
    'chest_gold_add': ('金箱 +1', 'Num4', 'gold', 1),
    'chest_gold_subtract': ('金箱 −1', 'Num5', 'gold', -1),
    'chest_purple_add': ('紫箱 +1', 'Num1', 'purple', 1),
    'chest_purple_subtract': ('紫箱 −1', 'Num2', 'purple', -1),
    'chest_red_add': ('红箱 +1', 'Num7', 'red', 1),
    'chest_red_subtract': ('红箱 −1', 'Num8', 'red', -1),
}
CHEST_HOTKEY_FIELDS = tuple(f'{action}_hotkey' for action in CHEST_ACTIONS)


class ChestCounter:
    def __init__(self):
        self.gold = 0
        self.purple = 0
        self.red = 0
        self._history: list[tuple[int, int, int]] = []

    @property
    def can_undo(self):
        return bool(self._history)

    def _set(self, gold, purple, red):
        if (gold, purple, red) == (self.gold, self.purple, self.red):
            return False
        self._history.append((self.gold, self.purple, self.red))
        self.gold, self.purple, self.red = gold, purple, red
        return True

    def apply(self, action):
        _, _, kind, delta = CHEST_ACTIONS[action]
        counts = dict(gold=self.gold, purple=self.purple, red=self.red)
        counts[kind] = max(0, counts[kind] + delta)
        return self._set(**counts)

    def reset(self):
        """Keep reset undoable so an accidental new-round click loses no counts."""
        return self._set(0, 0, 0)

    def undo(self):
        if not self._history:
            return False
        self.gold, self.purple, self.red = self._history.pop()
        return True
