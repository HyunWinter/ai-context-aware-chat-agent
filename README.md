# Context-Aware AI Support Agent (Proof of Concept)

An exploratory project studying how to implement persistent long-term memory in LLM applications. This proof-of-concept integrates Streamlit, OpenAI, and Mem0 (backed by Qdrant) to demonstrate how an agent can remember past interactions and user preferences across multiple sessions.

> **Note:** The application serves as a flexible harness for testing memory architectures. While currently configured with a Nike customer support persona for demonstration purposes, the domain and instructions can be easily swapped out.

## How it works

1. User enters their unique ID and types a message in the Streamlit UI.
2. The agent queries Qdrant via Mem0 to fetch relevant past context for that specific user.
3. OpenAI's gpt-4o (configurable) generates a response based on the system prompt, user query, and retrieved memory context.
4. The new query and response are saved back to Qdrant to update the user's persistent memory.

## Interesting bits

* **Memory persistence** - Uses `Mem0` on top of a local `Qdrant` vector database. If a user mentions a specific preference in one session, the agent remembers it in the next.
* **API compatibility bypass** - Mem0's default OpenAI LLM integration forces `max_tokens` which is deprecated for newer models and causes 400 errors. The code injects a custom LLM config with `max_tokens=None` to bypass this issue without waiting for upstream fixes.
* **Separation of concerns** - Clean split between UI (`Streamlit`), Data (`MemoryService`), and Logic (`CustomerSupportAgent`).
* **Session management** - Streamlit state automatically clears the chat history if a different User ID is entered in the sidebar to prevent context leaking.

## Setup

Requires Python 3.10+, Docker, and an OpenAI API key.

```bash
# 1. Start Qdrant vector database (must mount volume to persist data)
docker run -d -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key" > .env

# 4. Run the app
streamlit run app.py
```

## Files

* `app.py` - Core application logic, Mem0 integration, and Streamlit UI
* `.streamlit/config.toml` - UI tweaks (forces light mode, hides developer options)
* `requirements.txt` - Python dependencies including `mem0ai[nlp]` for BM25 hybrid search
* `customer_support_agent.py` - Original reference script (deprecated)

## License

MIT