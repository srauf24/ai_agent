from api.agent import handle_request

def test_queries():
    # Test cases
    queries = [
        "What is 25 * 48?",  # Calculator test
        "Who won the latest Super Bowl?",  # Web search test
        "What is Python programming?",  # Direct answer test
    ]

    for query in queries:
        print(f"\n\nTesting query: {query}")
        print("-" * 50)
        
        result = handle_request(query)
        print(f"Tool used: {result['tool_used']}")
        print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    print("Starting agent tests...")
    test_queries()