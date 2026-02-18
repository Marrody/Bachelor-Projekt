from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from ba_ragmas_chatbot.graph.state import AgentState
from ba_ragmas_chatbot.llm.factory import get_llm_for_agent
from ba_ragmas_chatbot.graph.utils import get_agent_config, get_task_config
from ba_ragmas_chatbot.tools.vectorstore import get_retriever
from ba_ragmas_chatbot.tools.search_tool import perform_web_search
from ba_ragmas_chatbot import logger_config

logger = logger_config.get_logger("GraphNodes")


def research_node(state: AgentState):
    """
    Fetches context from vector store AND/OR Web Search.
    """
    logger.info(f"RESEARCHER started for topic: {state['topic']}")

    agent_cfg = get_agent_config("researcher")
    task_cfg = get_task_config("research_task")
    current_date = datetime.now().strftime("%d. %B %Y")
    history_list = state.get("history", [])
    logger.info(f"History-length: {len(history_list)}")
    history_str = (
        "\n".join(history_list) if history_list else "No previous conversation."
    )
    context_text = ""
    retriever = get_retriever(k=4)

    if retriever and state.get("source_documents"):
        logger.info("🔍 Search local documents...")
        try:
            docs = retriever.invoke(state["topic"])
            if docs:
                logger.info(f"   ✅ {len(docs)} Document-Chunks found.")
                context_text += "=== LOCAL DOCUMENTS (HIGH TRUST) ===\n"
                context_text += "\n\n".join([d.page_content for d in docs])
                context_text += "\n====================================\n"
            else:
                logger.info("❌ Found no relevant documents.")
        except Exception as e:
            logger.error(f"⚠️ Retrieval failed: {e}")

    logger.info(f"🌍 Search web for: {state['topic']}")
    try:
        web_results = perform_web_search(state["topic"], max_results=3)
        if web_results:
            logger.info(f"✅ {len(web_results)} Web-results found.")
            context_text += "\n=== WEB SEARCH RESULTS (CHECK DATE) ===\n"
            context_text += "\n".join(
                [
                    f"Title: {r['title']}\nContent: {r['body']}\nSource: {r['href']}"
                    for r in web_results
                ]
            )
            context_text += "\n=======================================\n"
    except Exception as e:
        logger.error(f"⚠️ Web search failed: {e}")

    if not context_text:
        context_text = "NO DATA FOUND via RAG or Web. Use internal knowledge strictly."

    system_prompt = agent_cfg["role"].format(
        topic=state["topic"], language=state["language"]
    )
    system_prompt += f"\n\nCurrent Date: {current_date}"
    system_prompt += f"\n\nBackstory: {agent_cfg['backstory'].format(topic=state['topic'], current_date=current_date)}"
    user_prompt = task_cfg["description"].format(
        topic=state["topic"], history=history_str, current_date=current_date
    )
    user_prompt += f"\n\n### RESEARCH MATERIAL ###\n{context_text}"
    user_prompt += f"\n\nEXPECTED OUTPUT:\n{task_cfg['expected_output'].format(topic=state['topic'], language=state['language'])}"

    llm = get_llm_for_agent("researcher")

    logger.info("🤖 RESEARCHER started.")
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    logger.info(f"📝 RESEARCHER OUTPUT:\n{response.content[:500]}...\n(gekürzt)")

    return {
        "research_data": [response.content],
        "current_status": "Research completed.",
    }


def editor_node(state: AgentState):
    logger.info("🧠 EDITOR started.")

    agent_cfg = get_agent_config("editor")
    task_cfg = get_task_config("editor_task")
    research_summary = "\n".join(state.get("research_data", []))
    history_str = (
        "\n".join(state.get("history", [])) if state.get("history") else "None"
    )

    system_prompt = agent_cfg["role"].format(
        topic=state["topic"], language=state["language"]
    )
    user_prompt = task_cfg["description"].format(
        topic=state["topic"],
        length=state["target_len"],
        information_level=state["target_audience"],
        language_level=state["target_audience"],
        target_audience=state["target_audience"],
        tone=state["tone"],
        language=state["language"],
        additional_information=state["additional_info"],
        history=history_str,
    )
    user_prompt += f"\n\nRESEARCH MATERIAL:\n{research_summary}"
    user_prompt += f"\n\nEXPECTED OUTPUT:\n{task_cfg['expected_output'].format(topic=state['topic'], language=state['language'])}"

    llm = get_llm_for_agent("editor")
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    logger.info(f"📝 EDITOR OUTLINE:\n{response.content}")
    return {"outline": [response.content], "current_status": "Outline created."}


def writer_node(state: AgentState):
    logger.info("✍️ WRITER started.")

    agent_cfg = get_agent_config("writer")
    task_cfg = get_task_config("writer_task")
    outline_str = "\n".join(state.get("outline", []))
    history_str = (
        "\n".join(state.get("history", [])) if state.get("history") else "None"
    )

    system_prompt = agent_cfg["role"].format(
        topic=state["topic"], language=state["language"]
    )
    user_prompt = task_cfg["description"].format(
        topic=state["topic"],
        length=state["target_len"],
        information_level=state["target_audience"],
        language_level=state["target_audience"],
        target_audience=state["target_audience"],
        tone=state["tone"],
        language=state["language"],
        additional_information=state["additional_info"],
        history=history_str,
    )
    user_prompt += f"\n\nOUTLINE TO FOLLOW:\n{outline_str}"

    llm = get_llm_for_agent("writer")
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    logger.info(f"📝 WRITER DRAFT (first 200 characters): {response.content[:200]}...")
    return {"draft": response.content, "current_status": "Draft written."}


def fact_check_node(state: AgentState):
    logger.info("⚖️ FACT_CHECKER started.")
    agent_cfg = get_agent_config("fact_checker")
    task_cfg = get_task_config("fact_check_task")
    draft_text = state.get("draft", "")
    research_summary = "\n".join(state.get("research_data", []))
    system_prompt = agent_cfg["role"].format(
        topic=state["topic"], language=state["language"]
    )
    user_prompt = task_cfg["description"].format(topic=state["topic"], history="")
    user_prompt += f"\n\n--- RESEARCH BRIEFING (TRUE FACTS) ---\n{research_summary}"
    user_prompt += f"\n\n--- DRAFT TO CHECK ---\n{draft_text}"
    user_prompt += f"\n\nEXPECTED OUTPUT:\n{task_cfg['expected_output']}"
    llm = get_llm_for_agent("fact_checker")
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    logger.info("✅ Fact Check finished.")
    return {"draft": response.content, "current_status": "Fact check completed."}


def polisher_node(state: AgentState):
    logger.info("✨ POLISHER started.")

    agent_cfg = get_agent_config("polisher")
    task_cfg = get_task_config("polishing_task")
    draft_text = state.get("draft", "")
    system_prompt = agent_cfg["role"].format(
        topic=state["topic"], language=state["language"], tone=state["tone"]
    )
    user_prompt = task_cfg["description"].format(
        topic=state["topic"], tone=state["tone"], language=state["language"], history=""
    )
    user_prompt += f"\n\n--- TEXT TO POLISH ---\n{draft_text}"
    user_prompt += f"\n\nEXPECTED OUTPUT:\n{task_cfg['expected_output']}"
    llm = get_llm_for_agent("polisher")
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    logger.info("✅ Polishing finished.")
    return {"final_article": response.content, "current_status": "Polishing finished."}
