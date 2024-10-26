import os 
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
config = {"configurable": {"thread_id": "thread_1"}}


# Define the function that calls the model
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}

def getStarted():
    
    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # First query with memory test 
    query = "Hi! I'm Bob."
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()  # output contains all messages in state

    # Second query with memory test 
    query = "What's my name?"
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()



if __name__ == "__main__":
    getStarted()
