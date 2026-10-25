from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatOpenAI()
title_llm = ChatOpenAI()

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):

    messages = state['messages']
    res = llm.invoke(messages)
    return {'messages': [res]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

conn.execute("""
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMART KEY,
        title TEXT NOT NULL
    )
""")

conn.commit()

def save_chat_title(thread_id, title):
    conn.execute(
        """
    INSERT OR REPLACE INTO chat_titles (thread_id, title)
    VALUES(?, ?)
""",
    (str(thread_id), title)
    )
    conn.commit()

def retrieve_all_chat_titles():
    cursor = conn.execute(
        "SELECT thread_id, title from chat_titles"
    )
    return dict(cursor.fetchall())

# Checkpoint
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config['configurable']['thread_id'])
    return list(threads)


chatbot