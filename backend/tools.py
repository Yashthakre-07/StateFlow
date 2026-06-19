import requests
from pydantic import BaseModel, Field, field_validator
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from asteval import Interpreter
from .config import settings

# 1. Web Search Tool Input Validation Schema
class SearchInput(BaseModel):
    query: str = Field(..., description="The query to search the web for.")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        if len(v) > 100:
            raise ValueError("Query too long (max 100 characters).")
        return v

@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> str:
    """
    Search DuckDuckGo for the latest news and information.
    """
    search_run = DuckDuckGoSearchRun(region="us-en")
    return search_run.run(query)


# 2. Calculator Tool Input Validation Schema
class CalculatorInput(BaseModel):
    expression: str = Field(..., description="The mathematical expression to evaluate safely (e.g. '2548 * 92' or '(45 + 5) * 10').")

    @field_validator('expression')
    @classmethod
    def validate_expression(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Expression cannot be empty.")
        if len(v) > 200:
            raise ValueError("Expression too long (max 200 characters).")
        return v

@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression string using a sandboxed interpreter.
    """
    try:
        aeval = Interpreter()
        result = aeval(expression)
        if aeval.error:
            # Retrieve error message securely
            err_msg = str(aeval.error[0].get_error()[1]) if len(aeval.error[0].get_error()) > 1 else "Invalid mathematical expression"
            return {"error": err_msg}
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


# 3. Stock Price Tool Input Validation Schema
class StockInput(BaseModel):
    symbol: str = Field(..., description="The stock ticker symbol (e.g. 'AAPL', 'TSLA').")

    @field_validator('symbol')
    @classmethod
    def alpha_only(cls, v: str) -> str:
        v = v.strip()
        if not v.isalpha():
            raise ValueError('Ticker must contain letters only')
        if not (1 <= len(v) <= 5):
            raise ValueError('Ticker must be between 1 and 5 characters')
        return v.upper()

@tool("get_stock_price", args_schema=StockInput)
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given ticker symbol.
    """
    if not settings.alpha_vantage_key:
        return {"error": "Alpha Vantage API Key is missing. Please set the ALPHA_VANTAGE_KEY environment variable."}
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={settings.alpha_vantage_key}"
    try:
        r = requests.get(url)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# Export tools
tools = [web_search, get_stock_price, calculator]
