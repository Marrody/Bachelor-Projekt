from langchain_ollama import ChatOllama
from ba_ragmas_chatbot.graph.utils import load_yaml_config


def get_llm_for_agent(agent_name: str, temperature: float = 0.7):
    """
    Returns the configured LLM instance for a specific agent.
    """

    try:
        config = load_yaml_config("configs.yaml")

        if "chatbot" in config and "llm" in config["chatbot"]:
            model_name = config["chatbot"]["llm"].get(
                "name", "llama3.1:8b-instruct-q8_0"
            )
            base_url = config["chatbot"]["llm"].get("url", "http://localhost:11434")

        else:
            raw_model_name = config.get("agents", {}).get(
                "llm", "llama3.1:8b-instruct-q8_0"
            )
            model_name = raw_model_name.replace("ollama/", "")
            base_url = config.get("agents", {}).get("url", "http://localhost:11434")

    except Exception as e:
        print(f"⚠️ Config Load Error in Factory: {e}. Using defaults.")
        model_name = "llama3.1:8b-instruct-q8_0"
        base_url = "http://localhost:11434"

    if agent_name == "researcher":
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.1)

    elif agent_name == "writer":
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.7)

    elif agent_name == "fact_checker":
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.0)

    elif agent_name == "polisher":
        return ChatOllama(model=model_name, base_url=base_url, temperature=0.5)

    return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
