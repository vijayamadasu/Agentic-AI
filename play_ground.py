from phi.agent import Agent
from  phi.model.groq import Groq  #Groq offers blazing-fast API endpoints for large language models
from phi.tools.yfinance import YFinanceTools  #YFinanceTools enable an Agent to access stock data, financial information and more from Yahoo Finance.
from phi.tools.duckduckgo import DuckDuckGo
import openai
import phi

import os
from dotenv import load_dotenv

from phi.playground import Playground, serve_playground_app
#Load enviornment variable from .env file

load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

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


app=Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("Playground:app", reload=True)