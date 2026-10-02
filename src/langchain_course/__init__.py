from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch

# tavily = TavilyClient()  # Initialize Tavily client

# @tool
# def search(query: str) -> str:
#     """
#     Tool That search over internet
#     Args:
#         query: The query to search for
#         Returns:
#             The search results
#     """

#     print(f"Searching for: {query}")
#     # Simulate a search operation (replace with actual search logic)
#     return tavily.search(query = query)  # Use Tavily client to perform the search

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatOllama(model="qwen3.5:9b")
tools = [TavilySearch()]  # Add the TavilySearch tool to the list of tools
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
            {
                "messages": HumanMessage(
                    content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
                )
            }
        )
    print(result)


if __name__ == "__main__":
    main()
