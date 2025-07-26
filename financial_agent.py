from phi.agent import Agent
from  phi.model.groq import Groq  #Groq offers blazing-fast API endpoints for large language models
from phi.tools.yfinance import YFinanceTools  #YFinanceTools enable an Agent to access stock data, financial information and more from Yahoo Finance.
from phi.tools.duckduckgo import DuckDuckGo
import openai

import os
from dotenv import load_dotenv
load_dotenv()


openai.api_key=os.getenv("OPENAI_API_KEY")

## Web search Agent
# This agent collect the stock data from web needed to analyse 
web_search_agent =Agent(
    name="web_search_Agent",
    role="Search the web for information",
   model=Groq(id="llama-3.3-70b-versatile"),
    tools =[DuckDuckGo()],
    instructions=["Always include source"],
    show_tools_calls =True,
    markdown=True,


)


##Financial Agent
finance_agent =Agent(
    name ="Finance_AI_Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools =[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instruction =["Use tables to display the data"],
    show_tools_calls = True,
    markdown =True,
   
)


# Combine  both the above agents into single multi ai agent 

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    instructions= ["Always include the sources","Format your response using markdown and use tables to display data where possible."],
    model=Groq(id="llama-3.3-70b-versatile"),
    show_tool_calls=True,
    markdown=True
)
multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for Qualcomm", stream=True)
