import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def analyze_with_llm(text):
    """Analyzes text using OpenAI's API to extract meaningful product-related keywords."""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""Analyze this product review transcript and extract ONLY meaningful product-related keywords.
        Focus STRICTLY on:
        1. Product features (e.g., 'battery', 'screen', 'camera')
        2. Product components (e.g., 'processor', 'ram', 'storage')
        3. Product aspects (e.g., 'design', 'quality', 'performance')
        
        IMPORTANT RULES:
        - DO NOT include general words like 'better', 'how', 'its', 'like', 'more', 'new', 'good', 'bad'
        - DO NOT include verbs or adjectives
        - ONLY include nouns that are directly related to the product
        - If a word is mentioned positively, list it under POSITIVE KEYWORDS
        - If a word is mentioned negatively, list it under NEGATIVE KEYWORDS
        
        Format your response EXACTLY as follows:
        
        POSITIVE KEYWORDS:
        - [List ONLY product-related keywords that were positively mentioned]
        
        NEGATIVE KEYWORDS:
        - [List ONLY product-related keywords that were negatively mentioned]
        
        Transcript:
        {text}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a product review analyzer that extracts ONLY meaningful product-related keywords. You are strict about only including product features, components, and aspects."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        return None