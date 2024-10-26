import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence

from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import (
    SystemMessage,
    trim_messages,
    AIMessage,
    BaseMessage,
    HumanMessage,
)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
config = {"configurable": {"thread_id": "thread_1"}}

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Define the function that calls the model
def call_model(state: State):
    chain = prompt | llm
    # response = chain.invoke(state)

    trimmed_messages = trimmer.invoke(state["messages"])
    response = chain.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )


    return {"messages": response}


def getStarted():

    # Define a new graph
    # workflow = StateGraph(state_schema=MessagesState)
    workflow = StateGraph(state_schema=State)

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # # First query with memory test
    # query = "Hi! I'm Bob."
    # language = "Spanish"

    # input_messages = [HumanMessage(query)]

    # output = app.invoke({"messages": input_messages, "language": language}, config)
    # output["messages"][-1].pretty_print()  # output contains all messages in state

    # # Second query with memory test
    # query = "What's my name?"
    # input_messages = [HumanMessage(query)]
    # output = app.invoke({"messages": input_messages}, config)
    # output["messages"][-1].pretty_print()

    query = "Hi I'm Todd, please tell me a joke."
    language = "English"

    input_messages = [HumanMessage(query)]
    for chunk, metadata in app.stream(
        {"messages": input_messages, "language": language},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            print(chunk.content, end="|")




if __name__ == "__main__":
    getStarted()
    print("Completed")
