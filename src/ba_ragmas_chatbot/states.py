from enum import IntEnum
from typing import List


class S(IntEnum):
    """Conversation states for the blog configuration wizard."""

    CHAT = 0
    TOPIC = 1
    TASK = 2
    TOPIC_OR_TASK = 3
    WEBSITE = 4
    DOCUMENT = 5
    LENGTH = 6
    LEVEL = 7
    INFO = 8
    LANGUAGE = 9
    TONE = 10
    CONFIRM = 11
    ADDITIONAL = 12


STATE_FLOW: List[S] = [
    S.TOPIC_OR_TASK,
    S.TOPIC,
    S.TASK,
    S.WEBSITE,
    S.DOCUMENT,
    S.LENGTH,
    S.LEVEL,
    S.INFO,
    S.LANGUAGE,
    S.TONE,
    S.ADDITIONAL,
    S.CONFIRM,
]


def total_steps() -> int:
    return len(STATE_FLOW)


def step_index(state: S) -> int:
    """index of the state within STATE_FLOW."""
    return STATE_FLOW.index(state)


def next_state(state: S) -> S:
    """Return the next state. if already at the last, return the last again."""
    i = step_index(state)
    return STATE_FLOW[min(i + 1, len(STATE_FLOW) - 1)]


def prev_state(state: S) -> S:
    """Return the previous state. if already at the first, return the first again."""
    i = step_index(state)
    return STATE_FLOW[max(i - 1, 0)]


def first_state() -> S:
    return STATE_FLOW[0]


def last_state() -> S:
    return STATE_FLOW[-1]


def is_first(state: S) -> bool:
    return state == first_state()


def is_last(state: S) -> bool:
    return state == last_state()
