from langchain_ollama import ChatOllama
from ba_ragmas_chatbot.graph.utils import load_yaml_config


def get_llm_for_agent(agent_name: str, temperature: float = 0.7):
    """
    Returns the specialized LLM instance for a specific agent.
    """

    try:
        config = load_yaml_config("configs.yaml")

        if "chatbot" in config and "llm" in config["chatbot"]:
            default_model = config["chatbot"]["llm"].get("name", "llama3.1:8b")
            base_url = config["chatbot"]["llm"].get("url", "http://localhost:11434")
        else:

            raw_model = config.get("agents", {}).get("llm", "llama3.1:8b")
            default_model = raw_model.replace("ollama/", "")
            base_url = config.get("agents", {}).get("url", "http://localhost:11434")

    except Exception as e:
        print(f"⚠️ Config Load Error in Factory: {e}. Using hardcoded defaults.")
        default_model = "llama3.1:8b"
        base_url = "http://localhost:11434"

    if agent_name == "researcher":
        return ChatOllama(
            model="llama3.1:8b", base_url=base_url, temperature=0.1, keep_alive=0
        )

    elif agent_name == "editor":
        return ChatOllama(
            model="llama3.1:8b", base_url=base_url, temperature=0.6, keep_alive=0
        )

    elif agent_name == "writer":
        return ChatOllama(
            model="llama3.1:8b", base_url=base_url, temperature=0.7, keep_alive=0
        )

    elif agent_name == "fact_checker":
        return ChatOllama(
            model="llama3.1:8b", base_url=base_url, temperature=0.0, keep_alive=0
        )

    elif agent_name == "polisher":
        return ChatOllama(
            model="llama3.1:8b", base_url=base_url, temperature=0.5, keep_alive=0
        )

    print(f"Using default model '{default_model}' for agent '{agent_name}'")
    return ChatOllama(
        model=default_model, base_url=base_url, temperature=temperature, keep_alive=0
    )
