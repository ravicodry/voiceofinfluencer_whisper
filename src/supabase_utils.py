import os
from supabase import create_client
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def save_segment_product_analysis(analyzed_segments):
    """Saves the analyzed segments to Supabase."""
    try:
        for segment in analyzed_segments:
            # Convert segment to JSON-serializable format
            segment_data = {
                'video_url': segment['video_url'],
                'video_title': segment['video_title'],
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'text': segment['text'],
                'sentiment': segment['sentiment'],
                'keywords': json.dumps(segment['keywords']),
                'good_aspect': json.dumps(segment['good_aspect']),
                'bad_aspect': json.dumps(segment['bad_aspect']),
                'product_name': segment['product_name']
            }
            
            # Insert into Supabase
            supabase.table('product_reviews').insert(segment_data).execute()
            
    except Exception as e:
        print(f"Error saving to Supabase: {e}")

def load_segment_product_analysis():
    """Loads the analyzed segments from Supabase."""
    try:
        response = supabase.table('product_reviews').select("*").execute()
        segments = response.data
        
        # Convert JSON strings back to lists
        for segment in segments:
            segment['keywords'] = json.loads(segment['keywords'])
            segment['good_aspect'] = json.loads(segment['good_aspect'])
            segment['bad_aspect'] = json.loads(segment['bad_aspect'])
            
        return segments
    except Exception as e:
        print(f"Error loading from Supabase: {e}")
        return []

def get_product_statistics(product_name=None):
    """Gets statistics for a specific product or all products."""
    try:
        query = supabase.table('product_reviews').select("*")
        if product_name:
            query = query.eq('product_name', product_name)
        
        response = query.execute()
        segments = response.data
        
        # Calculate statistics
        total_reviews = len(segments)
        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        for segment in segments:
            sentiment = segment['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return {
            'total_reviews': total_reviews,
            'sentiment_distribution': sentiment_counts
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return None
