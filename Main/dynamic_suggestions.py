from dotenv import load_dotenv
from google import genai
import language_tool_python
from nltk.tokenize import sent_tokenize
import os

# Initialize the GenAI client 
load_dotenv()
client = genai.Client(api_key= os.getenv("OPENAI_API_KEY"))

def get_grammar_score(text):
    """
    Calculates grammar score and errors from the provided text.
    """
    tool = language_tool_python.LanguageTool('en-US')
    sentences = sent_tokenize(text)
    grammar_errors = sum(len(tool.check(sentence)) for sentence in sentences)
    grammar_score = max(0, 10 - ((grammar_errors / max(len(text.split()), 1)) * 100))
    return grammar_score, grammar_errors


def get_grammar_suggestions(grammar_score):
    prompt = f"""
    You are sitting in an interview and you're proficient in speaking english. Based on the grammar score provided, give:
    1. 3-4 grammar improvement suggestions tailored to the score.
    2. 2-3 general suggestions for improving formal speech.

    Grammar Score: {grammar_score}

    Suggestions:
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",  
            contents=prompt
        )
    
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    grammar_score = 6.5

    suggestions = get_grammar_suggestions(grammar_score)

    print("Dynamic Suggestions:")
    print(suggestions)