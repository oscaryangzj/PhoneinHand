"""按模型输出帧数进行因果滞回判决。"""

from __future__ import annotations

from collections import deque
from math import ceil

from . import config

UNKNOWN = -1
CONTACT = 0
FREE = 1


class FrameCountDebouncer:
    """强概率证据达到固定帧数比例后，确认 contact/free 状态。"""

    def __init__(self) -> None:
        self._state = UNKNOWN
        self._evidence: deque[int] = deque()
        self._frames_since_switch = 0

    @property
    def state(self) -> int:
        return self._state

    def reset(self) -> None:
        self._state = UNKNOWN
        self._evidence.clear()
        self._frames_since_switch = 0

    def update(self, probability_handheld: float) -> int:
        if probability_handheld >= config.HANDHELD_EVIDENCE_THRESHOLD:
            candidate = FREE
        elif probability_handheld <= config.NON_HANDHELD_EVIDENCE_THRESHOLD:
            candidate = CONTACT
        else:
            candidate = UNKNOWN

        self._frames_since_switch += 1
        self._evidence.append(candidate)
        if len(self._evidence) > config.CONTACT_TO_FREE_FRAMES:
            self._evidence.popleft()

        target = UNKNOWN
        if self._state == UNKNOWN:
            target = candidate
        elif self._state == CONTACT and candidate == FREE:
            target = FREE
        elif self._state == FREE and candidate == CONTACT:
            target = CONTACT

        if target == UNKNOWN:
            return self._state

        window_frames = (
            config.CONTACT_TO_FREE_FRAMES
            if target == FREE
            else config.FREE_TO_CONTACT_FRAMES
        )
        if self._has_evidence(target, window_frames):
            self._state = target
            self._evidence.clear()
            self._frames_since_switch = 0
        return self._state

    def _has_evidence(self, target: int, window_frames: int) -> bool:
        if self._frames_since_switch < window_frames or len(self._evidence) < window_frames:
            return False

        window = list(self._evidence)[-window_frames:]
        required_frames = ceil(window_frames * config.STABLE_EVIDENCE_RATIO)
        return window[-1] == target and sum(label == target for label in window) >= required_frames
