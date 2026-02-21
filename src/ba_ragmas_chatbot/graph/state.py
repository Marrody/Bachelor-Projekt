from typing import List, Optional, TypedDict, Any


class AgentState(TypedDict):
    """
    State of blogpost-workflow. Saves data, that is shared between the agents.
    """

    topic: str
    target_len: str
    language_level: str
    information_level: str
    language: str
    tone: str
    additional_info: str
    source_documents: List[str]
    research_data: List[str]
    outline: List[str]
    draft: str
    critique: Optional[str]
    final_article: str
    revision_count: int
    current_status: str
