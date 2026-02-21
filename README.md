# BaRagmasChatbot 2.0 (LangGraph Edition)

This repository contains the LangGraph-based version of the BaRagmasChatbot bachelor project: a local, retrieval‑augmented, multi‑agent system that can generate blog posts via a Telegram chatbot.

The system uses:
- **LangGraph** for explicit workflow orchestration 
- **Ollama** for local LLM inference and embeddings
- **Chroma** as local vector store (optional, only if documents are provided)

---
## Prerequisites

- Python **>= 3.10**
- Git
- A Telegram account
- GPU with at least 8GB VRAM

---
## 1) Get Ollama (Linux)

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama service (keep this running in a terminal):

```bash
ollama serve
```

### Download the required models

Open a second terminal and pull the models used by this project:

```bash
ollama pull llama3.1:8b-instruct-q8_0
ollama pull qwen2.5:7b-instruct-q5_k_m
ollama pull gemma2:9b-instruct-q5_k_m
ollama pull mxbai-embed-large
```


Verify installed models:

```bash
ollama list
```

---
## 2) Get a Telegram Bot Token (BotFather)

1. Open Telegram and search for **@BotFather**
2. Start the chat and run:
   - `/newbot`
3. Follow the prompts:
   - Choose a bot name (display name)
   - Choose a username ending in `bot` (must be unique)
4. BotFather will return a token that looks like:
   - `123456789:ABCDefGHIjkLMNopQRstuVWxyz`

Keep this token private.

---
## 3) Installation (Linux)

This project supports multiple installation approaches.

### Option A - pip + venv 

```bash
git clone <repository_url>
cd <project_folder>

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Option B - uv 

Install uv:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone <repository_url>
cd <project_folder>

uv sync
```

### Option C - Editable install 

```bash
git clone <repository_url>
cd <project_folder>

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

---
## 4) Configuration

Create a `.env` file in the repository root (or edit your existing one) and set:

```text
TELEGRAM_TOKEN=YOUR_TOKEN_FROM_BOTFATHER
```


---
## 5) Start the Bot

### Start Ollama (Terminal 1)

```bash
ollama serve
```

### Start the Telegram bot (Terminal 2)

From the project root:

```bash
source .venv/bin/activate  # only if you used venv
python -m ba_ragmas_chatbot.chatbot
```

You should see log output that the bot is running. Open your bot in Telegram and start chatting.

---
## 6) Stop the Bot

- Stop the Telegram bot: press **Ctrl + C** in Terminal 2
- Stop Ollama: press **Ctrl + C** in Terminal 1 (or stop the service if running in the background)

---
## Output

Generated blog posts are returned as:
- A Markdown-formatted Telegram message
- A downloadable `.md` file

---
## Notes on Local Data (documents/ and db/)

- `documents/` is created when users upload documents.
- `db/` is created when uploaded documents are indexed (Chroma vector store).

---
## License

Academic / Research Use

