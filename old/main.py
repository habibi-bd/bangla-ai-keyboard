import asyncio
from typing import TypedDict, Optional, Annotated

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# tools
from tools import (
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words,
    save_word_to_study_list
)

# --- Optional Anki Tools ---
from langchain_core.tools import tool

@tool
def create_anki_card(word: str, translation: str, deck_name: str):
    """
    Creates an Anki flashcard and saves it to local file.
    """
    entry = f"Deck: {deck_name} | Word: {word} | Translation: {translation}\n"
    with open("anki_deck.txt", "a", encoding="utf-8") as f:
        f.write(entry)
    return f"Added '{word}' to '{deck_name}'"

@tool
def create_anki_deck(deck_name: str):
    """
    Creates a new Anki deck (local placeholder).
    """
    return f"Deck '{deck_name}' ready."

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    source_language: Optional[str]
    number_of_words: Optional[int]
    word_difficulty: Optional[str]
    target_language: Optional[str]

# --- TOOL LIST ---
all_tools = [
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words,
    save_word_to_study_list,
    create_anki_deck,
    create_anki_card
]

# --- ASSISTANT NODE ---
def assistant(state: AgentState):

    sys_msg = SystemMessage(content="""
You are STRICTLY a tool-using agent.

RULES:
- You MUST use tools for ANY word generation
- NEVER answer from your own knowledge
- If user asks for words, ALWAYS call get_n_random_words
- If user asks translation, ALWAYS use translate_words tool

Do NOT guess words manually.
""")

   
    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    llm_with_tools = llm.bind_tools(all_tools)

    response = llm_with_tools.invoke([sys_msg] + state["messages"])

    return {
        "messages": [response],
        "source_language": state.get("source_language"),
        "number_of_words": state.get("number_of_words"),
        "word_difficulty": state.get("word_difficulty"),
        "target_language": state.get("target_language")
    }

# --- GRAPH ---
async def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(all_tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()

# --- MAIN ---
async def main():
    react_graph = await build_graph()

    user_prompt = """
CALL TOOLS ONLY.

Step 1: get 2 Spanish words using tool
Step 2: translate using tool
Step 3: save to Anki

DO NOT answer directly.
"""

    messages = [HumanMessage(content=user_prompt)]

    print(" Running LangGraph Agent with Llama3.2")

    result = await react_graph.ainvoke({
        "messages": messages,
        "source_language": None,
        "number_of_words": None,
        "word_difficulty": None,
        "target_language": None
    })

    print("\n--- FINAL OUTPUT ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())