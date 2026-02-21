from langchain_ollama import ChatOllama
from ba_ragmas_chatbot.graph.utils import get_model_config


def get_llm_for_agent(agent_name: str, temperature: float = 0.7):
    """
    Returns the specialized LLM instance for a specific agent.
    """
    config = get_model_config()
    base_url = config.get("base_url", "http://localhost:11434")
    logic_model = config.get("logic_model", "qwen2.5:7b-instruct-q5_k_m")
    creative_model = config.get("creative_model", "gemma2:9b-instruct-q5_k_m")

    if agent_name == "researcher":
        return ChatOllama(
            model=logic_model, base_url=base_url, temperature=0.0, keep_alive=0
        )

    elif agent_name == "editor":
        return ChatOllama(
            model=logic_model, base_url=base_url, temperature=0.2, keep_alive=0
        )

    elif agent_name == "writer":
        return ChatOllama(
            model=creative_model, base_url=base_url, temperature=0.7, keep_alive=0
        )

    elif agent_name == "fact_checker":
        return ChatOllama(
            model=logic_model, base_url=base_url, temperature=0.0, keep_alive=0
        )

    elif agent_name == "polisher":
        return ChatOllama(
            model=creative_model, base_url=base_url, temperature=0.6, keep_alive=0
        )

    return ChatOllama(
        model=logic_model, base_url=base_url, temperature=temperature, keep_alive=0
    )
