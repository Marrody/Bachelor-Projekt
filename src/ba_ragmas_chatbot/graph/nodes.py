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
    Distinguishes clearly between Local Docs (High Trust) and Web (Low Trust).
    """
    topic = state["topic"]
    logger.info(f"🕵️ RESEARCHER started for topic: {topic}")

    agent_cfg = get_agent_config("researcher")
    task_cfg = get_task_config("research_task")
    current_date = datetime.now().strftime("%d. %B %Y")

    local_context = ""
    retriever = get_retriever(k=4)
    has_documents = bool(state.get("source_documents"))

    if retriever and has_documents:
        logger.info("   🔍 Search local documents...")
        try:
            docs = retriever.invoke(topic)
            if docs:
                logger.info(f"   ✅ {len(docs)} Document-Chunks found.")
                local_context = "\n".join([f"- {d.page_content}" for d in docs])
            else:
                logger.info("   ❌ Found no relevant documents.")
        except Exception as e:
            logger.error(f"⚠️ Retrieval failed: {e}")

    web_context = ""
    logger.info(f"🌍 Search web for: {topic}")
    try:
        query_llm = get_llm_for_agent("researcher")
        query_prompt = f"Extract a concise 3-4 word search query for a web search engine from this topic: '{topic}'. Output ONLY the search query words, nothing else. Do not use quotation marks."

        search_query = (
            query_llm.invoke([HumanMessage(content=query_prompt)])
            .content.strip()
            .replace('"', "")
        )
        logger.info(f"🌍 Executing Web Search with optimized query: {search_query}")

        web_results = perform_web_search(search_query, max_results=3)
        if web_results:
            logger.info(f"✅ {len(web_results)} Web-results found.")
            web_context = "\n".join(
                [
                    f"Title: {r['title']}\nContent: {r['body']}\nSource: {r['href']}"
                    for r in web_results
                ]
            )
    except Exception as e:
        logger.error(f"⚠️ Web search failed: {e}")

    final_context = ""

    if local_context:
        final_context += f"=== 📂 LOCAL DOCUMENTS (PRIMARY SOURCE - HIGH TRUST) ===\n{local_context}\n\n"

    if web_context:
        final_context += f"=== 🌐 WEB SEARCH RESULTS (SECONDARY SOURCE - VERIFY FACTS) ===\n{web_context}\n\n"

    if not final_context:
        final_context = (
            "NO EXTERNAL DATA FOUND. Rely on internal knowledge but admit uncertainty."
        )

    system_prompt = agent_cfg["role"].format(topic=topic, language=state["language"])
    system_prompt += f"\n\nCurrent Date: {current_date}"
    system_prompt += f"\n\nBackstory: {agent_cfg['backstory'].format(topic=topic, current_date=current_date)}"

    user_prompt = task_cfg["description"].format(topic=topic, current_date=current_date)
    user_prompt += f"\n\n### AVAILABLE KNOWLEDGE ###\n{final_context}"
    user_prompt += "\n\nINSTRUCTION: Distinguish clearly between facts from Local Documents and Web Search in your briefing."
    user_prompt += f"\n\nEXPECTED OUTPUT:\n{task_cfg['expected_output'].format(topic=topic, language=state['language'])}"

    llm = get_llm_for_agent("researcher")
    logger.info("🤖 Researcher is thinking...")
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    logger.info(f"📝 RESEARCHER OUTPUT:\n{response.content[:500]}...\n(truncated)")

    return {
        "research_data": [response.content],
        "current_status": "Research completed.",
    }


def editor_node(state: AgentState):
    logger.info("🏗️ EDITOR started.")

    agent_cfg = get_agent_config("editor")
    task_cfg = get_task_config("editor_task")
    research_summary = "\n".join(state.get("research_data", []))

    system_prompt = agent_cfg["role"].format(
        topic=state["topic"], language=state["language"]
    )

    user_prompt = task_cfg["description"].format(
        topic=state["topic"],
        length=state["target_len"],
        information_level=state["information_level"],
        language_level=state["language_level"],
        tone=state["tone"],
        language=state["language"],
        additional_information=state["additional_info"],
    )
    user_prompt += f"\n\nRESEARCH MATERIAL:\n{research_summary}"
    user_prompt += f"\n\nEXPECTED OUTPUT:\n{task_cfg['expected_output']}"

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

    critique = state.get("critique") or ""

    system_prompt = agent_cfg["role"].format(
        topic=state["topic"], language=state["language"]
    )

    user_prompt = task_cfg["description"].format(
        topic=state["topic"],
        length=state["target_len"],
        information_level=state["information_level"],
        language_level=state["language_level"],
        tone=state["tone"],
        language=state["language"],
        additional_information=state["additional_info"],
    )
    user_prompt += f"\n\nOUTLINE TO FOLLOW:\n{outline_str}"

    if critique and "PASS" not in critique.upper():
        logger.warning(
            "⚠️ Writer has to rewrite draft because of alert from fact Checker!"
        )
        user_prompt += f"\n\n⚠️ YOUR PREVIOUS DRAFT HAD ERRORS. PLEASE FIX THEM BASED ON THIS CRITIQUE:\n{critique}\n\n--- PREVIOUS DRAFT ---\n{state.get('draft', '')}"

    llm = get_llm_for_agent("writer")
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    logger.info(f"📝 WRITER DRAFT (first 200 chars): {response.content[:200]}...")
    return {"draft": response.content, "current_status": "Draft written."}


def fact_check_node(state: AgentState):
    logger.info("⚖️ FACT CHECKER started.")

    agent_cfg = get_agent_config("fact_checker")
    task_cfg = get_task_config("fact_check_task")
    draft_text = state.get("draft", "")
    research_summary = "\n".join(state.get("research_data", []))
    rev_count = state.get("revision_count", 0)
    system_prompt = agent_cfg["role"].format(
        topic=state["topic"], language=state["language"]
    )
    user_prompt = task_cfg["description"].format(topic=state["topic"])
    user_prompt += f"\n\n--- RESEARCH BRIEFING (TRUE FACTS) ---\n{research_summary}"
    user_prompt += f"\n\n--- DRAFT TO CHECK ---\n{draft_text}"
    user_prompt += f"\n\nEXPECTED OUTPUT:\n{task_cfg['expected_output']}"

    llm = get_llm_for_agent("fact_checker")
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    critique = response.content.strip()
    logger.info(f"⚖️ Fact Check Result: {critique[:100]}...")

    return {
        "critique": critique,
        "revision_count": rev_count + 1,
        "current_status": f"Fact check completed (Revision {rev_count + 1}).",
    }


def polisher_node(state: AgentState):
    logger.info("✨ POLISHER started.")

    agent_cfg = get_agent_config("polisher")
    task_cfg = get_task_config("polishing_task")
    draft_text = state.get("draft", "")

    system_prompt = agent_cfg["role"].format(
        topic=state["topic"], language=state["language"], tone=state["tone"]
    )
    user_prompt = task_cfg["description"].format(
        topic=state["topic"], tone=state["tone"], language=state["language"]
    )
    user_prompt += f"\n\n--- TEXT TO POLISH ---\n{draft_text}"
    user_prompt += f"\n\nEXPECTED OUTPUT:\n{task_cfg['expected_output']}"
    user_prompt += (
        "\nIMPORTANT: Output ONLY the final article. No intro/outro conversation."
    )

    llm = get_llm_for_agent("polisher")
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    logger.info("✅ Polishing finished.")
    return {"final_article": response.content, "current_status": "Polishing finished."}
