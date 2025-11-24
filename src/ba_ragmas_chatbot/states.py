from enum import IntEnum


class S(IntEnum):
    """Conversation states for the blog configuration wizard."""

    TOPIC_OR_TASK = 0
    TOPIC = 1
    TASK = 2
    WEBSITE = 3
    DOCUMENT = 4
    LENGTH = 5
    LEVEL = 6
    INFO = 7
    LANGUAGE = 8
    TONE = 9
    ADDITIONAL = 10
    CONFIRM = 11


__all__ = [
    "S",
]
