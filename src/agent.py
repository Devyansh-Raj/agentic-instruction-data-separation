from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from .tools import read_mock_database, execute_system_command

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def get_llm(model_id: str):
    # Route to Google Gemini natively
    if "gemini" in model_id:
        print(f"Routing {model_id} via Google Gemini Natively")
        api_key = os.getenv("GEMINI_API_KEY", "dummy_gemini_key")
        return ChatGoogleGenerativeAI(model=model_id, temperature=0, api_key=api_key)
        
    # Route to OpenAI natively
    if "gpt" in model_id:
        print(f"Routing {model_id} via OpenAI Natively")
        api_key = os.getenv("OPENAI_API_KEY", "dummy_openai_key")
        return ChatOpenAI(model=model_id, temperature=0, api_key=api_key)
        
    # Route to Google Cloud vLLM cluster (Now Ollama VM)
    if ("qwen" in model_id.lower() or "vllm" in model_id.lower() or "llama" in model_id.lower() or "hermes" in model_id.lower() or "mistral" in model_id.lower()) and model_id not in ["llama3.1:latest", "qwen2.5:3b"]:
        print(f"Routing {model_id} via Google Cloud vLLM")
        vllm_url = os.getenv("VLLM_URL")
        if not vllm_url:
            raise RuntimeError("VLLM_URL not set in .env")
        return ChatOpenAI(
            model=model_id,
            temperature=0,
            api_key="ollama",
            base_url=vllm_url,
            max_retries=0,
            timeout=600.0,
        )
        
    # Fallback: Route to local Ollama
    print(f"Routing {model_id} via Local Ollama Fallback")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
    return ChatOpenAI(
        model=model_id,
        temperature=0,
        api_key="ollama",
        base_url=ollama_url
    )

def build_agent(model_id: str):
    llm = get_llm(model_id)
    tools = [read_mock_database, execute_system_command]
    llm_with_tools = llm.bind_tools(tools)
    
    def call_model(state: AgentState):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    def call_tools(state: AgentState):
        messages = state['messages']
        last_message = messages[-1]
        
        tool_messages = []
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "read_mock_database":
                result = read_mock_database.invoke(tool_call["args"])
            elif tool_call["name"] == "execute_system_command":
                result = execute_system_command.invoke(tool_call["args"])
            else:
                result = f"Error: Tool {tool_call['name']} not found."
                
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_call["name"]))
            
        return {"messages": tool_messages}
        
    def should_continue(state: AgentState):
        messages = state['messages']
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        return "continue"
        
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()
