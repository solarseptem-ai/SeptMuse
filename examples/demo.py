"""SeptMuse + Finna API 对话检索 Demo。

运行:
    PYTHONPATH=src python examples/demo.py
"""

from septmuse.llms.openai import OpenAILLM
from septmuse.embedders.openai import OpenAIEmbedder
from septmuse.configs.base import MemoryConfig
from septmuse.configs.database import DatabaseConfig
from septmuse.memory.main import Memory

BASE_URL = "https://www.finna.com.cn/v1"
LLM_API_KEY = "app-UQhuK4zkWCCIlngALxFUaVNU"
LLM_MODEL = "qwen3-8b"
EMBED_API_KEY = "app-t2XGtpLhHNYnUVa1tdV7uxEk"
EMBED_MODEL = "text-embedding-v1"

llm = OpenAILLM(api_key=LLM_API_KEY, model=LLM_MODEL, base_url=BASE_URL)
embedder = OpenAIEmbedder(api_key=EMBED_API_KEY, model=EMBED_MODEL, base_url=BASE_URL)
memory = Memory(config=MemoryConfig(database=DatabaseConfig(db_path="demo.db")), llm=llm, embedder=embedder)


def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    relevant_memories = memory.search(query=message, user_id=user_id, top_k=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories)

    system_prompt = f"You are a helpful AI with memory. Answer based on the user's past memories.\nUser Memories:\n{memories_str}"
    response = memory.llm.complete(system_prompt, message)

    memory.add(f"user: {message}\nassistant: {response}", user_id=user_id)
    return response


def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")


if __name__ == "__main__":
    main()