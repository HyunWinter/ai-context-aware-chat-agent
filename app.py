import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st
from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

# ==========================================
# 1. Configuration & Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class AppConfig:
    """Application configuration management."""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    app_id: str = "customer-support-app"

config = AppConfig()

# ==========================================
# 2. Constants & Prompts
# ==========================================
SYSTEM_PROMPT = """You are a friendly and knowledgeable Customer Support AI for Nike.
You assist customers with sneaker sizing, product recommendations, order tracking, return policies, and upcoming SNKRS releases.
Use the provided relevant past information to assist the customer effectively, accurately, and personally.
If the past information is not relevant to the current shopping query, you may ignore it."""


# ==========================================
# 3. Services (Data Access Layer)
# ==========================================
class MemoryService:
    """Handles all interactions with the Mem0 and Qdrant vector database."""
    
    def __init__(self, host: str, port: int, llm_model: str):
        mem0_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {"host": host, "port": port}
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "max_tokens": None, # Bypass max_tokens deprecation issue for gpt-4o
                }
            }
        }
        try:
            self.memory = Memory.from_config(mem0_config)
            logger.info("MemoryService initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize Mem0")
            raise RuntimeError(f"Memory initialization failed: {e}")

    def get_relevant_context(self, query: str, user_id: str) -> str:
        """Retrieves context string formatted from past relevant memories."""
        results = self.memory.search(query=query, filters={"user_id": user_id})
        
        if not results or "results" not in results:
            return ""
            
        memories = [m.get("memory") for m in results["results"] if m.get("memory")]
        if not memories:
            return ""
            
        context_str = "Relevant past information:\n"
        context_str += "\n".join(f"- {mem}" for mem in memories)
        return context_str

    def save_interaction(self, query: str, answer: str, user_id: str, app_id: str) -> None:
        """Saves a query-answer pair into the user's memory."""
        metadata = {"app_id": app_id}
        self.memory.add(query, user_id=user_id, metadata={**metadata, "role": "user"})
        self.memory.add(answer, user_id=user_id, metadata={**metadata, "role": "assistant"})

    def get_all_memories(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetches all raw memories for a specific user."""
        try:
            return self.memory.get_all(filters={"user_id": user_id})
        except Exception as e:
            logger.error(f"Error fetching memories for user {user_id}: {e}")
            return None


# ==========================================
# 4. Agent (Business Logic Layer)
# ==========================================
class CustomerSupportAgent:
    """Orchestrates LLM calls and memory integration."""
    
    def __init__(self, cfg: AppConfig, memory_service: MemoryService):
        self.config = cfg
        self.memory_service = memory_service
        self.client = OpenAI(api_key=cfg.openai_api_key)

    def process_query(self, query: str, user_id: str) -> str:
        """Processes a user query by retrieving context and calling the LLM."""
        try:
            # 1. Context Retrieval
            context = self.memory_service.get_relevant_context(query, user_id)
            
            # 2. Prompt Construction
            full_prompt = f"{context}\n\nCustomer: {query}\nSupport Agent:" if context else f"Customer: {query}\nSupport Agent:"
            
            # 3. LLM Generation
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7 # Slight randomness for conversational tone
            )
            answer = response.choices[0].message.content
            
            # 4. Save Interaction
            self.memory_service.save_interaction(query, answer, user_id, self.config.app_id)
            
            return answer
            
        except Exception as e:
            logger.exception("Error processing user query")
            return "I apologize, but I encountered a system error while processing your request. Please try again."


# ==========================================
# 5. UI (Presentation Layer - Streamlit)
# ==========================================
def init_session_state() -> None:
    """Initializes Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "current_customer" not in st.session_state:
        st.session_state.current_customer = None

def main():
    st.set_page_config(
        page_title="Support AI",
        page_icon="👟",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    init_session_state()

    # --- Header ---
    st.title("Nike Customer Support AI")
    st.caption("A context-aware shopping assistant for Nike members and sneakerheads")

    if not config.openai_api_key:
        st.error("🚨 `OPENAI_API_KEY` is missing. Please check your `.env` file.")
        st.stop()

    # --- Dependency Injection & Initialization ---
    try:
        if st.session_state.agent is None:
            memory_svc = MemoryService(config.qdrant_host, config.qdrant_port, config.openai_model)
            st.session_state.agent = CustomerSupportAgent(config, memory_svc)
    except RuntimeError as e:
        st.error(f"🚨 Initialization Error: {e}")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("👤 Member Context")
        customer_id = st.text_input("Enter Member ID", placeholder="e.g., nike-member-77")
        
        # Reset chat if a new customer logs in
        if customer_id and st.session_state.current_customer != customer_id:
            st.session_state.messages = []
            st.session_state.current_customer = customer_id
            
        st.divider()
        
        # Memory Viewer
        if st.button("🔍 View Memory Info", use_container_width=True):
            if not customer_id:
                st.warning("Please enter a Customer ID first.")
            else:
                with st.spinner("Fetching memories..."):
                    memories = st.session_state.agent.memory_service.get_all_memories(customer_id)
                    if memories and "results" in memories and memories["results"]:
                        st.subheader("Stored Memories")
                        for m in memories["results"]:
                            st.info(m.get("memory", ""))
                    else:
                        st.info("No memories found for this customer.")

    # --- Main Chat Interface ---
    if not customer_id:
        st.info("👋 Please enter your **Member ID** in the sidebar to start chatting.")
        st.stop()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if query := st.chat_input("Ask me about sneakers, sizing, or your recent orders..."):
        # Add to UI state immediately
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Process and respond
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = st.session_state.agent.process_query(query, customer_id)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
