import os
from dotenv import load_dotenv
import requests

# Load .env file from the same directory as the script
load_dotenv(dotenv_path='.env')

def test_huggingface_api():
    # Get the token from environment variables
    api_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
    
    # Debug: Print all env variables (safely)
    print("Environment variables:")
    for key in os.environ:
        if 'token' in key.lower() or 'key' in key.lower():
            print(f"{key}: {'*' * 8}")
        else:
            print(f"{key}: {os.environ[key]}")
    
    # Rest of your function remains the same
    if api_token:
        print(f"Token loaded (first 8 chars): {api_token[:8]}...")
    else:
        print("No token found in environment variables!")
        return False

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Testing with a simple prompt
    data = {
        "inputs": "What is artificial intelligence?"
    }
    
    api_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    
    try:
        print("Making API request...")
        response = requests.post(api_url, headers=headers, json=data)
        print(f"Response status code: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        print("Success! API Response:")
        print(result)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        return False

if __name__ == "__main__":
    print("Testing Hugging Face API connection...")
    test_huggingface_api()