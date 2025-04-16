from dotenv import load_dotenv
from google import genai
import os

# Initialize the GenAI client with your API key
load_dotenv()
client = genai.Client(api_key= os.getenv("OPENAI_API_KEY"))

# Function to get grammar improvement suggestions
def get_grammar_suggestions(grammar_score):
    # Prompt for grammar suggestions based on the score
    prompt = f"""
    You are sitting in an interview and you're proficient in speaking english. Based on the grammar score provided, give:
    1. 3-4 grammar improvement suggestions tailored to the score.
    2. 2-3 general suggestions for improving formal speech.

    Grammar Score: {grammar_score}

    Suggestions:
    """
    try:
        # Call Google GenAI to generate content
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Use the appropriate GenAI model
            contents=prompt
        )
        # Extract and return the suggestions
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    # Example grammar score
    grammar_score = 6.5

    # Get suggestions
    suggestions = get_grammar_suggestions(grammar_score)

    # Print the suggestions
    print("Dynamic Suggestions:")
    print(suggestions)