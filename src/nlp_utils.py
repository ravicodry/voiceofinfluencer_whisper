import nltk
from textblob import TextBlob
import re
import warnings
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from src.llm_utils import analyze_with_llm

# Suppress the specific warning
warnings.filterwarnings("ignore", category=SyntaxWarning, module="textblob")

# Download required NLTK data
try:
    nltk.data.find('punkt')
    nltk.data.find('averaged_perceptron_tagger')
    nltk.data.find('stopwords')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Extended stop words list
STOP_WORDS = {
    # Common English stop words
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of',
    # Additional common words to filter
    'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must', 'its', 'it', 'they', 'them', 'their',
    'there', 'here', 'where', 'when', 'why', 'how', 'what', 'which', 'who', 'whom',
    'whose', 'if', 'then', 'else', 'when', 'while', 'though', 'although', 'because',
    'since', 'until', 'unless', 'whether', 'while', 'whereas', 'wherever',
    # Informal words
    'gonna', 'wanna', 'gotta', 'ya', 'yeah', 'ok', 'okay', 'hey', 'hi', 'hello',
    # Common adjectives that don't add value
    'new', 'old', 'good', 'bad', 'big', 'small', 'great', 'nice', 'well', 'better',
    'best', 'worst', 'many', 'much', 'more', 'most', 'few', 'little', 'less', 'least',
    # Common adverbs
    'very', 'really', 'quite', 'rather', 'too', 'so', 'just', 'only', 'even', 'still',
    'yet', 'already', 'also', 'again', 'ever', 'never', 'always', 'often', 'sometimes',
    'usually', 'generally', 'normally', 'typically', 'probably', 'possibly', 'maybe',
    'perhaps', 'definitely', 'certainly', 'absolutely', 'totally', 'completely'
}

# Product-related keywords to keep
PRODUCT_KEYWORDS = {
    # Hardware components
    'battery', 'screen', 'display', 'camera', 'processor', 'ram', 'storage', 'memory',
    'chip', 'gpu', 'cpu', 'motherboard', 'keyboard', 'mouse', 'speaker', 'microphone',
    'port', 'usb', 'hdmi', 'jack', 'connector', 'cable', 'charger', 'adapter',
    
    # Product features
    'design', 'quality', 'performance', 'speed', 'power', 'efficiency', 'durability',
    'reliability', 'stability', 'compatibility', 'connectivity', 'wireless', 'bluetooth',
    'wifi', 'network', 'security', 'privacy', 'update', 'upgrade', 'version',
    
    # Product types
    'phone', 'laptop', 'tablet', 'computer', 'gadget', 'device', 'product', 'model',
    'brand', 'company', 'manufacturer', 'series', 'line', 'generation',
    
    # Technical terms
    'specs', 'specification', 'feature', 'function', 'capability', 'capacity',
    'resolution', 'pixel', 'megapixel', 'refresh', 'rate', 'frequency', 'bandwidth',
    'latency', 'response', 'time', 'battery', 'life', 'runtime', 'capacity',
    
    # Product aspects
    'price', 'cost', 'value', 'budget', 'premium', 'luxury', 'affordable',
    'warranty', 'guarantee', 'support', 'service', 'maintenance', 'repair',
    'replacement', 'refund', 'return', 'policy'
}

# Custom filter for irrelevant words
IRRELEVANT_WORDS = {
    'better', 'how', 'its', 'like', 'more', 'new', 'old', 'good', 'bad', 'great', 'nice', 'well',
    'very', 'really', 'quite', 'rather', 'too', 'so', 'just', 'only', 'even', 'still', 'yet',
    'already', 'also', 'again', 'ever', 'never', 'always', 'often', 'sometimes', 'usually',
    'generally', 'normally', 'typically', 'probably', 'possibly', 'maybe', 'perhaps', 'definitely',
    'certainly', 'absolutely', 'totally', 'completely', 'gonna', 'wanna', 'gotta', 'ya', 'yeah',
    'ok', 'okay', 'hey', 'hi', 'hello', 'this', 'that', 'these', 'those', 'here', 'there', 'where',
    'when', 'why', 'what', 'which', 'who', 'whom', 'whose', 'if', 'then', 'else', 'when', 'while',
    'though', 'although', 'because', 'since', 'until', 'unless', 'whether', 'while', 'whereas',
    'wherever', 'it', 'they', 'them', 'their', 'there', 'here', 'where', 'when', 'why', 'how',
    'what', 'which', 'who', 'whom', 'whose', 'if', 'then', 'else', 'when', 'while', 'though',
    'although', 'because', 'since', 'until', 'unless', 'whether', 'while', 'whereas', 'wherever'
}

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get English stop words
stop_words = set(stopwords.words('english'))

def analyze_sentiment(text):
    """Analyzes the sentiment of a given text using TextBlob."""
    try:
        analysis = TextBlob(text)
        # Get polarity (-1 to 1)
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "neutral"

def extract_keywords(text):
    """Extracts product-related keywords using NLP techniques."""
    try:
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Get words and their POS tags
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        
        # Define product-related POS tags (nouns only, no pronouns)
        product_tags = {'NN', 'NNS', 'NNP', 'NNPS'}  # Only nouns
        
        # Extract only nouns, lemmatize them, and filter out stop words and irrelevant words
        keywords = []
        for word, tag in pos_tags:
            if (tag in product_tags and 
                word not in stop_words and 
                word not in IRRELEVANT_WORDS and 
                len(word) > 2):
                # Only keep words that are either in PRODUCT_KEYWORDS or are meaningful nouns
                if word in PRODUCT_KEYWORDS or (tag in {'NNP', 'NNPS'} and len(word) > 3):
                    lemmatized_word = lemmatizer.lemmatize(word)
                    keywords.append(lemmatized_word)
        
        # Remove duplicates and sort by frequency
        keyword_counts = Counter(keywords)
        sorted_keywords = [word for word, count in keyword_counts.most_common()]
        
        return sorted_keywords
    except Exception as e:
        print(f"Error in keyword extraction: {e}")
        return []

def analyze_transcript_keywords(segments):
    """Analyzes keywords from the entire transcript, considering sentiment context."""
    try:
        # Combine all text and analyze as one
        all_text = ' '.join([seg['text'] for seg in segments])
        
        # Get all keywords from the entire transcript
        all_keywords = extract_keywords(all_text)
        
        # Create sentiment context for each keyword
        keyword_sentiment = {}
        for keyword in all_keywords:
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            # Look for the keyword in each segment
            for segment in segments:
                if keyword in segment['text'].lower():
                    if segment['sentiment'] == 'positive':
                        positive_count += 1
                    elif segment['sentiment'] == 'negative':
                        negative_count += 1
                    else:
                        neutral_count += 1
            
            # Calculate sentiment score (-1 to 1)
            total = positive_count + negative_count + neutral_count
            if total > 0:
                sentiment_score = (positive_count - negative_count) / total
                keyword_sentiment[keyword] = sentiment_score
        
        # Separate keywords into positive and negative based on sentiment score
        positive_keywords = [k for k, v in keyword_sentiment.items() if v > 0.1]
        negative_keywords = [k for k, v in keyword_sentiment.items() if v < -0.1]
        
        # Sort by absolute sentiment score
        positive_keywords.sort(key=lambda x: abs(keyword_sentiment[x]), reverse=True)
        negative_keywords.sort(key=lambda x: abs(keyword_sentiment[x]), reverse=True)
        
        return {
            'positive_keywords': positive_keywords[:10],  # Top 10 positive keywords
            'negative_keywords': negative_keywords[:10],  # Top 10 negative keywords
            'keyword_sentiment': keyword_sentiment
        }
    except Exception as e:
        print(f"Error analyzing transcript keywords: {e}")
        return {'positive_keywords': [], 'negative_keywords': [], 'keyword_sentiment': {}}

def generate_summary(segments):
    """Generates a comprehensive summary of the video analysis using LLM."""
    try:
        # Combine all text into one complete transcript
        complete_transcript = ' '.join([seg['text'] for seg in segments])
        
        # Get overall sentiment
        sentiments = [seg['sentiment'] for seg in segments]
        sentiment_counts = Counter(sentiments)
        overall_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        
        # Use LLM to analyze the complete transcript
        llm_analysis = analyze_with_llm(complete_transcript)
        
        # Extract keywords from LLM analysis
        positive_keywords = []
        negative_keywords = []
        
        if llm_analysis:
            # Parse LLM response to extract positive and negative keywords
            try:
                # Split LLM response into sections
                sections = llm_analysis.split('\n\n')
                for section in sections:
                    if 'POSITIVE KEYWORDS:' in section:
                        # Extract keywords from positive section
                        keywords = [word.strip('- ').strip() for word in section.split('\n') 
                                  if word.strip() and not word.startswith('POSITIVE KEYWORDS:')]
                        positive_keywords.extend(keywords)
                    elif 'NEGATIVE KEYWORDS:' in section:
                        # Extract keywords from negative section
                        keywords = [word.strip('- ').strip() for word in section.split('\n') 
                                  if word.strip() and not word.startswith('NEGATIVE KEYWORDS:')]
                        negative_keywords.extend(keywords)
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
        
        # Generate summary
        summary = {
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': dict(sentiment_counts),
            'positive_keywords': positive_keywords[:10],  # Top 10 positive keywords
            'negative_keywords': negative_keywords[:10],  # Top 10 negative keywords
            'total_segments': len(segments),
            'summary_text': f"""
            Overall Analysis:
            - The video has an overall {overall_sentiment} sentiment
            - Analyzed {len(segments)} segments
            
            Top Product Features (Positive):
            {chr(10).join(['• ' + keyword for keyword in positive_keywords[:5]])}
            
            Top Product Features (Negative):
            {chr(10).join(['• ' + keyword for keyword in negative_keywords[:5]])}
            """
        }
        
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None