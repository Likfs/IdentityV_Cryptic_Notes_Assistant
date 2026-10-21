"""Manual per-run chest counts, independent of map and projection sessions."""
from __future__ import annotations


CHEST_ACTIONS = {
    'chest_gold_add': ('金箱 +1', 'F5', 'gold', 1),
    'chest_gold_subtract': ('金箱 −1', 'F6', 'gold', -1),
    'chest_purple_add': ('紫箱 +1', 'F7', 'purple', 1),
    'chest_purple_subtract': ('紫箱 −1', 'F9', 'purple', -1),
}
CHEST_HOTKEY_FIELDS = tuple(f'{action}_hotkey' for action in CHEST_ACTIONS)


class ChestCounter:
    def __init__(self):
        self.gold = 0
        self.purple = 0
        self._history: list[tuple[int, int]] = []

    @property
    def can_undo(self):
        return bool(self._history)

    def _set(self, gold, purple):
        if (gold, purple) == (self.gold, self.purple):
            return False
        self._history.append((self.gold, self.purple))
        self.gold, self.purple = gold, purple
        return True

    def apply(self, action):
        _, _, kind, delta = CHEST_ACTIONS[action]
        return self._set(max(0, self.gold + delta) if kind == 'gold' else self.gold,
                         max(0, self.purple + delta) if kind == 'purple' else self.purple)

    def reset(self):
        """Keep reset undoable so an accidental new-round click loses no counts."""
        return self._set(0, 0)

    def undo(self):
        if not self._history:
            return False
        self.gold, self.purple = self._history.pop()
        return True
