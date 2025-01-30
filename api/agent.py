import os
import json
import re
from typing import Tuple, Optional
import requests
from duckduckgo_search import ddg

HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
MODEL_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

def calculator(expression: str) -> str:
    try:
        # Using eval with basic security measures
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

def web_search(query: str, max_results: int = 3) -> str:
    try:
        results = ddg(query, max_results=max_results)
        if not results:
            return "No results found."
        return "\n".join(f"- {r['title']}\n  {r['body']}\n  Link: {r['href']}" for r in results)
    except Exception as e:
        return f"Error performing search: {str(e)}"

def query_huggingface(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {"inputs": prompt}
    
    try:
        response = requests.post(MODEL_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            return result[0]
        return str(result)
    except Exception as e:
        return f"Error querying model: {str(e)}"

def parse_tool_request(response: str) -> Tuple[Optional[str], Optional[str]]:
    match = re.search(r"CallTool:\s*(calculator|search)\((.*?)\)", response)
    if match:
        return match.group(1), match.group(2).strip().strip('"').strip("'")
    return None, None

def create_agent_prompt(user_query: str) -> str:
    return f"""You are an AI with access to tools:
1) Calculator: "CallTool: calculator(expression)"
2) Search: "CallTool: search(query)"

Answer directly when possible. Otherwise, request a tool.
User Query: {user_query}"""

def agent(user_query: str) -> str:
    # First interaction with the model
    initial_prompt = create_agent_prompt(user_query)
    first_response = query_huggingface(initial_prompt)
    
    # Check if tool usage is requested
    tool, arg = parse_tool_request(first_response)
    
    if tool:
        # Execute the requested tool
        tool_result = calculator(arg) if tool == "calculator" else web_search(arg)
        
        # Get final answer with tool results
        final_prompt = f"""Previous response: {first_response}
Tool result: {tool_result}
Now provide a final answer based on the tool results."""
        
        return query_huggingface(final_prompt)
    
    return first_response

def handler(request):
    try:
        body = request.get_json()
        user_query = body.get("query", "")
        if not user_query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Query parameter is required"})
            }
        
        response = agent(user_query)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"answer": response})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }