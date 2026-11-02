"""The agent decides for itself when a recipe stage is done, from what it reads and what it did: pixels only.

Items show on the status bar, so a counted stage, a tool or a first item is a reading. A placed table doesn't show
once it is out of view, so "placed" = the agent played the place action and an item count fell. A reading must hold
for `hold` steps, so one misread doesn't derail the chain. See docs/DESIGN_RECIPE.md."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from jepa_game_agent.envs.base import Frame
from jepa_game_agent.evaluation.runner import ChainStage


class StageTracker:
    def __init__(
        self,
        stages: Sequence[ChainStage],
        read: Callable[[Frame], Mapping[str, float]],
        place_actions: Mapping[str, int],
        hold: int = 2,
    ) -> None:
        if hold < 1:
            raise ValueError("hold must be >= 1")
        for st in stages:
            if st.name.startswith("place_"):
                if st.name not in place_actions:
                    raise ValueError(f"no action known for {st.name!r}")
            elif not st.name.startswith(("collect_", "make_")):
                raise ValueError(f"can't sense {st.name!r}: only items and placing show")
        self.stages, self.read, self.place_actions, self.hold = list(stages), read, place_actions, hold
        self.stage, self._streak = 0, 0
        self._before: dict[str, int] = {}
        self._spent: dict[str, int] | None = None  # the counts just before a place action made one fall

    def start(self, frame: Frame) -> None:
        self._before = self._counts(frame)

    def _counts(self, frame: Frame) -> dict[str, int]:
        return {k: round(v) for k, v in self.read(frame).items()}

    def _done(self, st: ChainStage, now: Mapping[str, int], action: int) -> bool:
        if st.name.startswith("place_"):
            if action == self.place_actions[st.name] and any(n < self._before.get(i, 0) for i, n in now.items()):
                self._spent = dict(self._before)
            return self._spent is not None and any(n < self._spent.get(i, 0) for i, n in now.items())
        if st.item is not None:
            return now.get(st.item, 0) >= st.count
        return now.get(st.name.split("_", 1)[1], 0) >= 1

    def update(self, frame: Frame, action: int) -> int:
        """After playing `action` and seeing `frame`: how many stages are done, by the agent's own sense."""
        now = self._counts(frame)
        if self.stage < len(self.stages):
            self._streak = self._streak + 1 if self._done(self.stages[self.stage], now, action) else 0
            if self._streak >= self.hold:
                self.stage, self._streak, self._spent = self.stage + 1, 0, None
        self._before = now
        return self.stage
