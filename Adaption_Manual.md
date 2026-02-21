# Adaptation Manual – LangGraph Version

One of the primary design goals of this project is adaptability. The system architecture deliberately separates **configuration**, **workflow logic**, **agent behaviour**, **model selection**, **retrieval (RAG)**, and **interface**. This manual explains not only *what* can be adapted, but also *how* to implement typical modifications in a stable way.

---
## Quick Orientation

Key adaptation surfaces (most stable → most invasive):

1. **Prompts & templates** (`config/agents.yaml`, `config/tasks.yaml`)
2. **Global settings** (`config/configs.yaml`, `.env`)
3. **Workflow topology & routing** (`graph/workflow.py`)
4. **Node logic / state schema** (`graph/nodes.py`)
5. **Model factory & provider switching** (`llm/factory.py`)
6. **Retrieval pipeline** (`tools/vectorstore.py`, `tools/search_tool.py`)
7. **Telegram UX / conversation states** (`chatbot.py`, `states.py`)


---
## 1) Configuration Adaptation (config Folder)

The `config/` folder is the declarative control layer. It enables changes without touching executable code and is the preferred place to start. `factory.py` fetches the models automatically.

### 1.1 `config/configs.yaml` - Global Runtime Settings

Purpose:
- Default models (LLM + embeddings)


#### How to adapt 

**Model swap:**

```yaml
llm_model: llama3.1:8b-instruct-q4_K_M
embedding_model: mxbai-embed-large
```

---
### 1.2 `config/agents.yaml` - Agent Roles, Persona, Constraints

Purpose:
- Defines each agent's identity and behavioural constraints through a system prompt or role prompt.

Typical agents:
- Researcher
- Editor
- Writer
- FactChecker
- Polisher

#### How to adapt 

**Example: Introduce citation style instructions (Polisher):**

```yaml
polisher:
  system_prompt: |
    If sources are provided in the briefing, add inline citations like [1], [2].
    Do not invent sources.
```

---
### 1.3 `config/tasks.yaml` - Step Outputs & Deliverables

Purpose:
- Specifies what each step must produce (structure, checklists, formatting constraints).

#### How to adapt

**Make the editor output a strict outline format:**

```yaml
editor_task:
  description: |
    Create an outline with numbered H2 sections and bullet points per section.
    Include supporting evidence placeholders.
```

---
## 2) Workflow Adaptation (LangGraph)

The workflow is defined in:
- `graph/workflow.py`

Nodes are processing steps, edges define execution order. LangGraph enables conditional routing (revision loops).

### 2.1 Modifying the agent flow 

#### How to adapt 

**Add a new node** (example: SEO step after polishing):

```python
workflow.add_node("seo", seo_node)
workflow.add_edge("polisher", "seo")
workflow.add_edge("seo", END)
```

**Remove a node**:
- delete `add_node(...)`
- reconnect the edges so the graph remains connected

### 2.2 Changing revision logic 

The revision loop is controlled by conditional edges from the FactChecker node.

Possible changes:
- Increase max revision count
- Adjust PASS/FAIL criteria

#### How to adapt 

**Increase revision count**:
- Identify where `revision_count`  is incremented and compared.


---
## 3) Agent Behaviour Adaptation 

Agent logic is implemented in:
- `graph/nodes.py`

Each node encapsulates one role:
- `research_node`
- `editor_node`
- `writer_node`
- `fact_check_node`
- `polisher_node`

### 3.1 The shared state (how information flows)

LangGraph nodes typically:
- read from `state` (inputs produced by earlier nodes)
- write new keys back into `state`

#### How to adapt 

**Add a new state field** (example: `citations`):
1. In `research_node`, collect citations and store:
   - `state["citations"] = ...`
2. Update downstream prompts to include citations.
3. In `polisher_node`, enforce citation formatting.


### 3.2 Writer improvements (how rewrites are implemented)

Typical rewrite pattern:
- include `critique` and `previous_draft` in the prompt
- instruct the writer to apply only necessary edits

#### How to adapt 

Add 'diff-style' rewrite discipline:
- instruct the writer to keep sections unchanged unless flagged
- or to produce a short change log

---
## 4) Model Adaptation (`llm/factory.py`)

Model configuration is handled via:
- `llm/factory.py`

Purpose:
- Role-specific specialization (logic vs creative)

### 4.1 Changing LLM models (Ollama)

Download new model:

```bash
ollama pull <model_name>
```

Then change it in `config/configs.yaml`

#### How to adapt 

- change `temperature` for the agents flexiblity
- change `keep_alive` to controll, how long the model stays loaded in memory

### 4.2 Using non-Ollama models

To switch providers:
1. Replace `Ollama` initialization with the desired provider wrapper
2. Ensure output formats remain compatible with downstream nodes



---
## 5) RAG Adaptation

Retrieval components are defined in:
- `tools/vectorstore.py`
- `tools/search_tool.py`

### 5.1 Vector store adaptation (Chroma)

Typical parameters to tune:
- `chunk_size`
- `chunk_overlap`
- top-k retrieval

#### How to adapt 

- Smaller chunks improve precision for factual queries
- Larger chunks improve coherence for long-form writing

A common tuning path:

1. Increase overlap if key facts get split
2. Increase `k` if answers miss important context

### 5.2 Web search adaptation

The search tool typically:
- queries a search provider (DuckDuckGo)
- filters results by domain / filetype

#### How to adapt

- Extend blacklist/whitelist
- prefer authoritative domains (e.g., `.edu`, official docs)
- add a 'recency' bias if the topic is time-sensitive


---
## 6) Telegram Chatbot Adaptation

Conversation logic is defined in:
- `chatbot.py`

To adapt interaction flow:
1. Modify conversation states
2. Adjust question sequence
3. Update handlers and transitions

#### How to adapt 

- Add new configuration parameters by:
  - adding a new state (`states.py`)
  - updating the state machine transitions
  - passing the new field into the workflow inputs

- Change UI style:
  - replace free-text input with inline buttons


