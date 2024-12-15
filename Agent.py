import os
import operator

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    
    
class Agent:
    
    def __init__(self, model: ChatOpenAI, tools: list, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node('llm', self.call_openai)
        graph.add_node('action', self.take_action)
        graph.add_conditional_edges('llm', self.exists_action, {True: 'action', False: END})
        graph.add_edge('action', 'llm')
        graph.set_entry_point('llm')
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        
        
    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return { 'messages' : [message] }
    
    def exists_action(self, state: AgentState):
        results = state['messages'][-1]
        return len(results.tool_calls) > 0
    
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"\n\nБи энэ үйлдлийг хийхээр шийдлээ💡: {t}\n\n")
            if not t['name'] in self.tools:
                print(f"...bad tool name...\nretry...")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("\n\nЗа, би үйлдлээ дуусчихлаа, одоо юу хийхээ бодож цэгцэлье 🤔!\n\n")
        return { 'messages' : results }
    
    

def main():
    _ = load_dotenv()

    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
    
    tool = TavilySearchResults(max_results=4)
    
    prompt = """You are a smart research assistant that translates between Mongolian and English. Use in English the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    
    model = ChatOpenAI(model='gpt-4o', api_key=OPENAI_KEY)
    bot = Agent(model, [tool], system=prompt)
    
    text_query = input("Та юу мэдмээр байна: ")
    
    messages = [HumanMessage(content=text_query)]
    results = bot.graph.invoke({"messages": messages})
    
    for result in results['messages']:
        print(result.content)
        print("\n\n")

if __name__ == "__main__":
    main()