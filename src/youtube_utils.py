from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, VideoUnavailable, TranscriptsDisabled
import re
import requests
import os
from dotenv import load_dotenv
from functools import lru_cache
import time
import logging
import json
from datetime import datetime, timedelta

load_dotenv()  # Load environment variables from .env

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache for storing successful transcript retrievals
transcript_cache = {}
CACHE_DURATION = timedelta(hours=24)  # Cache duration of 24 hours

def extract_video_id(youtube_url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, youtube_url)
    if match:
        return match.group(1)
    return None

def get_transcript(video_url, max_retries=3):
    video_id = extract_video_id(video_url)
    if not video_id:
        return None, "❌ Invalid YouTube URL"

    # Check cache first
    if video_id in transcript_cache:
        cache_entry = transcript_cache[video_id]
        if datetime.now() - cache_entry['timestamp'] < CACHE_DURATION:
            return cache_entry['transcript'], None

    # Create a persistent session with proper headers
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }
    session.headers.update(headers)

    def try_get_transcript(method="standard", cookies=None):
        try:
            if method == "standard":
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                return transcript, None
            elif method == "any_language":
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                return transcript, None
            elif method == "with_cookies":
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'], cookies=cookies)
                return transcript, None
            elif method == "with_cookies_any_language":
                transcript = YouTubeTranscriptApi.get_transcript(video_id, cookies=cookies)
                return transcript, None
        except NoTranscriptFound:
            return None, "NoTranscriptFound"
        except VideoUnavailable:
            return None, "VideoUnavailable"
        except TranscriptsDisabled:
            return None, "TranscriptsDisabled"
        except Exception as e:
            logger.error(f"Error in try_get_transcript: {str(e)}")
            return None, str(e)

    # Try different methods with retries
    methods = [
        ("standard", None),
        ("any_language", None),
    ]

    # Get fresh cookies and session data
    try:
        # First visit YouTube homepage
        session.get("https://www.youtube.com")
        time.sleep(1)  # Small delay
        
        # Then visit the specific video
        session.get(f"https://www.youtube.com/watch?v={video_id}")
        time.sleep(1)  # Small delay
        
        cookies = session.cookies.get_dict()
        formatted_cookies = ";".join([f"{k}={v}" for k, v in cookies.items()])
        methods.extend([
            ("with_cookies", formatted_cookies),
            ("with_cookies_any_language", formatted_cookies)
        ])
    except Exception as e:
        logger.warning(f"Failed to get cookies: {str(e)}")

    # Try each method with exponential backoff
    for method, cookies in methods:
        for attempt in range(max_retries):
            logger.info(f"Attempt {attempt + 1}/{max_retries} using method: {method}")
            
            # Exponential backoff
            if attempt > 0:
                time.sleep(2 ** attempt)
            
            transcript, error = try_get_transcript(method, cookies)
            
            if transcript:
                # Cache the successful result
                transcript_cache[video_id] = {
                    'transcript': transcript,
                    'timestamp': datetime.now()
                }
                return transcript, None
            
            if error == "VideoUnavailable":
                return None, "❌ This video is unavailable or private."
            elif error == "TranscriptsDisabled":
                return None, "❌ Transcripts are disabled for this video."
            
            if error != "NoTranscriptFound":
                logger.warning(f"Error on attempt {attempt + 1}: {error}")
                continue

    return None, "❌ No transcript available for this video. Please try a different video."

@lru_cache(maxsize=100)
def get_video_details(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return None, "❌ Invalid YouTube URL"

    try:
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            return None, "❌ YouTube API key not found in environment variables."
        
        url = f"https://youtube.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}"
        response = requests.get(url)
        data = response.json()

        # Add logging to see the API response
        logger.info(f"YouTube API Response: {data}")

        if "error" in data:
            return None, f"❌ YouTube API Error: {data['error']['message']}"

        if not data.get("items"):
            return None, "❌ Video details not found."

        item = data["items"][0]
        # Log the statistics part specifically
        logger.info(f"Video Statistics: {item.get('statistics', {})}")
        
        # Calculate engagement rate (likes + comments) / views
        view_count = int(item["statistics"].get("viewCount", 0))
        like_count = int(item["statistics"].get("likeCount", 0))
        comment_count = int(item["statistics"].get("commentCount", 0))
        
        engagement_rate = 0
        if view_count > 0:
            engagement_rate = ((like_count + comment_count) / view_count) * 100
        
        # Determine if trending (high engagement rate and views)
        is_trending = engagement_rate > 5 and view_count > 10000
        
        details = {
            "title": item["snippet"]["title"],
            "view_count": view_count,
            "like_count": like_count,
            "comment_count": comment_count,
            "publish_date": item["snippet"]["publishedAt"],
            "engagement_rate": round(engagement_rate, 2),
            "is_trending": is_trending,
            "timestamp": time.time()  # Add timestamp for cache invalidation
        }
        return details, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        return None, f"❌ Network error fetching video details: {str(e)}"
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        return None, f"❌ Error fetching video details: {str(e)}"