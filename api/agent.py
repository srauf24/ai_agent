import os
import json
import re
from typing import Tuple, Optional
import requests
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

def query_model(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    data = {"inputs": prompt}
    api_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', str(result[0]))
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def calculator(expression: str) -> str:
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

def web_search(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        return "\n".join(f"- {r['title']}\n  {r['body']}\n  Link: {r['link']}" for r in results)
    except Exception as e:
        return f"Error performing search: {str(e)}"  # Fixed missing quotation

def parse_tool_request(response: str) -> Tuple[Optional[str], Optional[str]]:
    match = re.search(r"CallTool:\s*(calculator|search)\((.*?)\)", response)
    if match:
        return match.group(1), match.group(2).strip().strip('"\'')
    return None, None

def handle_request(query: str) -> dict:
    system_prompt = """You are a precise AI assistant that MUST respond in ONE of these THREE formats ONLY.
DO NOT explain, just respond in one of these formats:

1. CallTool: calculator(2+2)
2. CallTool: search(exact search query)
3. Direct Answer: your answer

For your current question about "{query}", choose ONE format and respond EXACTLY like the examples.
If asking about current events, sports, news, or facts, you MUST use: CallTool: search(query)

Response:"""
    
    # First interaction with the model
    first_response = query_model(system_prompt.format(query=query))
    
    # Clean up the response - take only what's after "Response:"
    first_response = first_response.split("Response:")[-1].strip()
    first_response = re.sub(r'<[^>]+>', '', first_response)
    print(f"Cleaned response: {first_response}")
    
    # Check if it's a direct answer
    if first_response.startswith("Direct Answer:"):
        return {"answer": first_response[14:].strip(), "tool_used": None}
    
    # Check for tool usage
    tool, arg = parse_tool_request(first_response)
    print(f"Detected tool: {tool}, arg: {arg}")
    
    if tool:
        # Execute requested tool
        tool_result = calculator(arg) if tool == "calculator" else web_search(arg)
        
        # Get final answer with strict formatting
        final_prompt = f"""Based on this information: {tool_result}

Provide a single, clear answer to: {query}
- No explanations
- No technical details
- Just the answer"""
        
        final_response = query_model(final_prompt)
        return {"answer": final_response, "tool_used": tool}
    
    # If no proper format was detected, force a direct answer
    return {"answer": first_response, "tool_used": None}

def handler(request):
    try:
        body = request.get_json()
        query = body.get("query", "")
        if not query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Query parameter is required"})
            }
        
        result = handle_request(query)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

if __name__ == "__main__":
    while True:
        try:
            # Get user input
            user_query = input("\nEnter your question (or 'exit' to quit): ")
            
            # Check for exit command
            if user_query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            # Process the query
            result = handle_request(user_query)
            
            # Print the result
            print("\nAnswer:", result['answer'])
            if result['tool_used']:
                print("Tool used:", result['tool_used'])
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")