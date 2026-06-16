import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
