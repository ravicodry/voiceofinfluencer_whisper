# Voice of Influencer (SIEL) - YouTube Product Review Analyzer

A powerful tool that analyzes YouTube product review videos to extract insights, sentiment, and key features using AI and natural language processing.

## Features

- **Multiple Transcription Methods**
  - YouTube API integration
  - Whisper AI transcription (local/cloud)
  - Support for multiple Whisper model sizes

- **Advanced Analysis**
  - Sentiment analysis of product reviews
  - Keyword extraction and frequency analysis
  - Product comparison capabilities
  - Engagement metrics analysis

- **Interactive Dashboard**
  - Real-time video analysis
  - Product comparison visualization
  - Keyword frequency charts
  - Video metadata display

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- YouTube API key (for YouTube API transcription method)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voice-of-influencer.git
cd voice-of-influencer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
   - MacOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

## Configuration

1. Create a `.env` file in the project root:
```
YOUTUBE_API_KEY=your_youtube_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter a YouTube video URL and product name to analyze

## Project Structure

```
voice-of-influencer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── README.md             # Project documentation
└── src/
    ├── youtube_utils.py  # YouTube API integration
    ├── whisper_utils.py  # Whisper transcription
    ├── nlp_utils.py      # NLP and sentiment analysis
    ├── llm_utils.py      # LLM integration
    └── storage_utils.py  # Data storage utilities
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Whisper for transcription capabilities
- Streamlit for the web interface
- YouTube Data API for video metadata
