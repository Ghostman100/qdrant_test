import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio, MCPServerSSE


URL = "212.41.9.143"
provider = OpenAIProvider(
    base_url=f"http://{URL}:9998/v1",
    api_key="EMPTY",
)


agent_model = OpenAIChatModel("Qwen/Qwen3-4B-Thinking-2507-FP8", provider=provider)

# MCP сервер Qdrant
qdrant_server = MCPServerSSE(f"http://{URL}:6339/sse")

# Создаем агента
agent = Agent(
    model=agent_model,
    toolsets=[qdrant_server],
    system_prompt=(
        "Ты агент с доступом к MCP-серверу Qdrant.\n"
        "Когда пользователь просит найти предложения по смыслу, "
        "используй инструмент Qdrant для семантического поиска.\n"
        "Выводи только найденные фразы и короткие пояснения.\n"
    ),
)

async def run_async(prompt: str) -> str:
    async with agent:
        result = await agent.run(prompt)
        return result.output

if __name__ == "__main__":
    query = "Найди предложение про дуб."
    print(asyncio.run(run_async(query)))