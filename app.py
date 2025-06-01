import streamlit as st
import pandas as pd
from src.youtube_utils import get_transcript, get_video_details
from src.llm_utils import analyze_with_llm
from src.storage_utils import save_segment_product_analysis, load_segment_product_analysis
from src.nlp_utils import analyze_sentiment
from collections import Counter
from src.whisper_utils import download_audio, convert_to_wav, load_whisper_model, transcribe_audio, cleanup_temp_files
from src.trend_analysis import show_trend_analysis
import subprocess

# Set page config
st.set_page_config(page_title="Voice of influencer(SIEL)", layout="wide")

# Create sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Video Analysis", "Trend Analysis"])

if page == "Video Analysis":
    st.title("🎥 YouTube Product Review Analyzer")
    
    # Input fields
    video_url = st.text_input("Enter YouTube Video URL:")
    st.sidebar.header("Product to Analyze")
    user_defined_product = st.sidebar.text_input("Enter the Product Name:")

    # Add transcription method selection
    transcription_method = st.selectbox(
        "Select Transcription Method:",
        ("YouTube API", "Whisper (Cloud/Local)")
    )

    # Add Whisper model selection if Whisper is chosen
    whisper_model_size = None
    if transcription_method == "Whisper (Cloud/Local)":
        whisper_model_size = st.selectbox(
            "Select Whisper Model Size:",
            ("tiny", "small", "base", "medium", "large"),
            index=2
        )

    # Cache functions
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_cached_transcript(video_url):
        return get_transcript(video_url)

    @st.cache_data(ttl=3600)
    def get_cached_video_details(video_url):
        return get_video_details(video_url)

    @st.cache_data(ttl=3600)
    def get_cached_sentiment(text):
        return analyze_sentiment(text)

    def detect_comparison_products(product_name):
        """Detect if the product name contains a comparison and return list of products."""
        # Common comparison separators
        separators = [' vs ', ' versus ', ' compared to ', ' comparison ']
        
        # Check if any separator exists in the product name
        for separator in separators:
            if separator in product_name.lower():
                # Split the product name and clean up each product name
                products = [p.strip() for p in product_name.split(separator)]
                if len(products) > 1:
                    return products
        
        # If no comparison detected, return single product
        return [product_name]

    def analyze_with_llm_focused(transcript, product_name):
        """Analyze transcript with a more focused prompt for emphasized keywords."""
        # Check if this is a comparison video
        products = detect_comparison_products(product_name)
        
        if len(products) > 1:
            # This is a comparison video
            prompt = f"""
            Analyze this product comparison video between {', '.join(products)}. For each product:
            1. Extract features and aspects that were:
               - Explicitly mentioned multiple times
               - Emphasized with strong positive or negative sentiment
               - Specifically discussed in detail
            
            Format your response as:
            
            SUMMARY:
            [Provide a concise summary of the comparison]
            
            {products[0]}:
            POSITIVE KEYWORDS:
            - keyword1 (with brief context of emphasis)
            - keyword2 (with brief context of emphasis)
            
            NEGATIVE KEYWORDS:
            - keyword1 (with brief context of emphasis)
            - keyword2 (with brief context of emphasis)
            
            {products[1]}:
            POSITIVE KEYWORDS:
            - keyword1 (with brief context of emphasis)
            - keyword2 (with brief context of emphasis)
            
            NEGATIVE KEYWORDS:
            - keyword1 (with brief context of emphasis)
            - keyword2 (with brief context of emphasis)
            
            Transcript:
            {transcript}
            """
        else:
            # Single product analysis
            prompt = f"""
            Analyze this product review transcript for {product_name}. Focus ONLY on features and aspects that were:
            1. Explicitly mentioned multiple times
            2. Emphasized with strong positive or negative sentiment
            3. Specifically discussed in detail
            
            Extract ONLY the most emphasized keywords that were actually discussed in the video.
            Ignore general mentions or passing references.
            
            Format your response as:
            
            SUMMARY:
            [Provide a concise summary of the main points discussed in the video]
            
            POSITIVE KEYWORDS:
            - keyword1 (with brief context of emphasis)
            - keyword2 (with brief context of emphasis)
            
            NEGATIVE KEYWORDS:
            - keyword1 (with brief context of emphasis)
            - keyword2 (with brief context of emphasis)
            
            Transcript:
            {transcript}
            """
        return analyze_with_llm(prompt)

    def display_video_analysis(video_title, video_details, llm_analysis, positive_freq, negative_freq, is_comparison=False, other_product_freq=None):
        """Display video analysis in the requested format."""
        st.subheader("📝 Video Analysis")
        
        # Display video details
        st.write(f"**Title:** {video_title}")
        
        # Format and display video metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            view_count = video_details.get('view_count')
            st.write(f"**Views:** {view_count:,}" if isinstance(view_count, (int, float)) else f"**Views:** {view_count}")
        
        with col2:
            like_count = video_details.get('like_count')
            st.write(f"**Likes:** {like_count:,}" if isinstance(like_count, (int, float)) else f"**Likes:** {like_count}")
        
        with col3:
            comment_count = video_details.get('comment_count')
            st.write(f"**Comments:** {comment_count:,}" if isinstance(comment_count, (int, float)) else f"**Comments:** {comment_count}")
        
        # Display publish date and engagement metrics
        col1, col2 = st.columns(2)
        
        with col1:
            publish_date = video_details.get('publish_date')
            if publish_date:
                # Convert ISO format to readable date
                from datetime import datetime
                date_obj = datetime.strptime(publish_date, "%Y-%m-%dT%H:%M:%SZ")
                formatted_date = date_obj.strftime("%B %d, %Y")
                st.write(f"**Published:** {formatted_date}")
        
        with col2:
            engagement_rate = video_details.get('engagement_rate')
            if engagement_rate is not None:
                st.write(f"**Engagement Rate:** {engagement_rate}%")
        
        # Display trending status
        if video_details.get('is_trending'):
            st.success("🔥 This video is trending!")
        
        # Extract and display summary
        if llm_analysis:
            summary_section = llm_analysis.split('POSITIVE KEYWORDS:')[0].replace('SUMMARY:', '').strip()
            st.write("**Summary:**")
            st.write(summary_section)
        
        # Display positive keywords with frequency > 1
        if positive_freq:
            st.subheader("Positive Keywords (Count > 1)")
            pos_data = {k: v for k, v in positive_freq.items() if v > 1}
            if pos_data:
                if is_comparison and other_product_freq:
                    # Create comparison DataFrame
                    all_keywords = set(pos_data.keys()) | set(other_product_freq.keys())
                    comparison_data = {
                        'Keyword': list(all_keywords),
                        'Current Product': [pos_data.get(k, 0) for k in all_keywords],
                        'Other Product': [other_product_freq.get(k, 0) for k in all_keywords]
                    }
                    comparison_df = pd.DataFrame(comparison_data).sort_values('Current Product', ascending=False)
                    
                    # Display side-by-side bar chart
                    st.bar_chart(comparison_df.set_index('Keyword'))
                    # Display comparison table
                    st.dataframe(comparison_df)
                else:
                    pos_df = pd.DataFrame({
                        'Keyword': list(pos_data.keys()),
                        'Count': list(pos_data.values())
                    }).sort_values('Count', ascending=False)
                    
                    # Display bar chart
                    st.bar_chart(pos_df.set_index('Keyword')['Count'])
                    # Display table
                    st.dataframe(pos_df)
            else:
                st.write("No positive keywords with count > 1")
        
        # Display negative keywords with frequency > 1
        if negative_freq:
            st.subheader("Negative Keywords (Count > 1)")
            neg_data = {k: v for k, v in negative_freq.items() if v > 1}
            if neg_data:
                if is_comparison and other_product_freq:
                    # Create comparison DataFrame
                    all_keywords = set(neg_data.keys()) | set(other_product_freq.keys())
                    comparison_data = {
                        'Keyword': list(all_keywords),
                        'Current Product': [neg_data.get(k, 0) for k in all_keywords],
                        'Other Product': [other_product_freq.get(k, 0) for k in all_keywords]
                    }
                    comparison_df = pd.DataFrame(comparison_data).sort_values('Current Product', ascending=False)
                    
                    # Display side-by-side bar chart
                    st.bar_chart(comparison_df.set_index('Keyword'))
                    # Display comparison table
                    st.dataframe(comparison_df)
                else:
                    neg_df = pd.DataFrame({
                        'Keyword': list(neg_data.keys()),
                        'Count': list(neg_data.values())
                    }).sort_values('Count', ascending=False)
                    
                    # Display bar chart
                    st.bar_chart(neg_df.set_index('Keyword')['Count'])
                    # Display table
                    st.dataframe(neg_df)
            else:
                st.write("No negative keywords with count > 1")

    # Main analysis function
    if st.button("Analyze"):
        if not video_url:
            st.error("Please enter a valid YouTube URL.")
        elif not user_defined_product:
            st.error("Please enter the name of the product you want to analyze in the sidebar.")
        else:
            transcript_list = None
            error = None
            if transcription_method == "YouTube API":
                with st.spinner("Fetching transcript using YouTube API..."):
                    transcript_list, error = get_transcript(video_url)
            else:
                # Whisper-based transcription
                in_file = wav_file = None
                try:
                    with st.spinner("Downloading audio from YouTube..."):
                        in_file = download_audio(video_url)
                    with st.spinner("Converting to WAV format..."):
                        wav_file = convert_to_wav(in_file)
                    with st.spinner(f"Loading Whisper model ({whisper_model_size})..."):
                        model = load_whisper_model(whisper_model_size)
                    with st.spinner("Transcribing audio with Whisper..."):
                        transcript_text = transcribe_audio(model, wav_file)
                    # Split transcript into segments (simulate YouTube API format)
                    transcript_list = [{
                        'text': transcript_text,
                        'start': 0,
                        'duration': 0
                    }]
                except subprocess.CalledProcessError as e:
                    error = f"FFmpeg error during conversion: {e}"
                except Exception as e:
                    error = f"Error: {e}"
                finally:
                    cleanup_temp_files()

            if error:
                st.error(error)
            elif not transcript_list:
                st.error("No transcript available for this video.")
            else:
                # Get video details
                details, error_details = get_cached_video_details(video_url)
                if error_details:
                    st.error(f"Error fetching video details: {error_details}")
                    details = {}
                video_title = details.get('title', 'N/A')

                # Combine all segments into one complete transcript
                complete_transcript = ' '.join([seg['text'] for seg in transcript_list])
                
                # Analyze the complete transcript
                with st.spinner("Analyzing transcript..."):
                    # Get overall sentiment
                    overall_sentiment = get_cached_sentiment(complete_transcript)
                    
                    # Get keywords using focused LLM analysis
                    llm_analysis = analyze_with_llm_focused(complete_transcript, user_defined_product)
                    
                    # Check if this is a comparison video
                    products = detect_comparison_products(user_defined_product)
                    
                    if len(products) > 1:
                        # Handle comparison analysis
                        analysis_results = []
                        for i, product in enumerate(products):
                            # Extract product-specific sections from LLM analysis
                            product_section = llm_analysis.split(f"{product}:")[1].split("POSITIVE KEYWORDS:")[0] if f"{product}:" in llm_analysis else ""
                            
                            # Extract positive and negative keywords
                            positive_keywords = []
                            negative_keywords = []
                            
                            if "POSITIVE KEYWORDS:" in product_section:
                                pos_section = product_section.split("POSITIVE KEYWORDS:")[1].split("NEGATIVE KEYWORDS:")[0]
                                for line in pos_section.split('\n'):
                                    if line.strip() and not line.startswith('POSITIVE KEYWORDS:'):
                                        # Extract the keyword part before any parentheses
                                        keyword = line.strip('- ').strip().split('(')[0].strip()
                                        # Clean up the keyword (remove any remaining punctuation)
                                        keyword = ''.join(c for c in keyword if c.isalnum() or c.isspace())
                                        if keyword:  # Only add non-empty keywords
                                            positive_keywords.append(keyword)
                            
                            if "NEGATIVE KEYWORDS:" in product_section:
                                neg_section = product_section.split("NEGATIVE KEYWORDS:")[1]
                                for line in neg_section.split('\n'):
                                    if line.strip() and not line.startswith('NEGATIVE KEYWORDS:'):
                                        # Extract the keyword part before any parentheses
                                        keyword = line.strip('- ').strip().split('(')[0].strip()
                                        # Clean up the keyword (remove any remaining punctuation)
                                        keyword = ''.join(c for c in keyword if c.isalnum() or c.isspace())
                                        if keyword:  # Only add non-empty keywords
                                            negative_keywords.append(keyword)
                            
                            # Count keyword frequencies
                            positive_freq = {}
                            negative_freq = {}
                            
                            # Convert transcript to lowercase for case-insensitive matching
                            transcript_lower = complete_transcript.lower()
                            
                            for keyword in positive_keywords:
                                # Convert keyword to lowercase for case-insensitive matching
                                keyword_lower = keyword.lower()
                                # Count occurrences using case-insensitive matching
                                count = transcript_lower.count(keyword_lower)
                                if count > 0:  # Only add keywords that appear at least once
                                    positive_freq[keyword] = count
                            
                            for keyword in negative_keywords:
                                # Convert keyword to lowercase for case-insensitive matching
                                keyword_lower = keyword.lower()
                                # Count occurrences using case-insensitive matching
                                count = transcript_lower.count(keyword_lower)
                                if count > 0:  # Only add keywords that appear at least once
                                    negative_freq[keyword] = count
                            
                            # Save analysis for this product
                            analysis_result = {
                                'video_url': video_url,
                                'video_title': video_title,
                                'video_details': details,
                                'product_name': product.strip(),
                                'overall_sentiment': overall_sentiment,
                                'positive_keywords': positive_keywords,
                                'negative_keywords': negative_keywords,
                                'positive_frequencies': positive_freq,
                                'negative_frequencies': negative_freq,
                                'summary': llm_analysis
                            }
                            analysis_results.append(analysis_result)
                        
                        # Save all analysis results
                        save_segment_product_analysis(analysis_results)
                        st.success("Analysis complete and saved! ✅")
                        
                        # Display results for each product with comparison
                        for i, result in enumerate(analysis_results):
                            st.subheader(f"Analysis for {result['product_name']}")
                            # Get the other product's frequencies for comparison
                            other_product_freq = None
                            if i == 0 and len(analysis_results) > 1:
                                other_product_freq = analysis_results[1]['positive_frequencies']
                            elif i == 1 and len(analysis_results) > 0:
                                other_product_freq = analysis_results[0]['positive_frequencies']
                            
                            display_video_analysis(
                                result['video_title'],
                                result['video_details'],
                                result['summary'],
                                result['positive_frequencies'],
                                result['negative_frequencies'],
                                is_comparison=True,
                                other_product_freq=other_product_freq
                            )
                    else:
                        # Handle single product analysis (existing code)
                        positive_keywords = []
                        negative_keywords = []
                        
                        if llm_analysis:
                            try:
                                # Split LLM response into sections
                                sections = llm_analysis.split('\n\n')
                                for section in sections:
                                    if 'POSITIVE KEYWORDS:' in section:
                                        # Extract keywords
                                        for line in section.split('\n'):
                                            if line.strip() and not line.startswith('POSITIVE KEYWORDS:'):
                                                # Extract the keyword part before any parentheses
                                                keyword = line.strip('- ').strip().split('(')[0].strip()
                                                # Clean up the keyword (remove any remaining punctuation)
                                                keyword = ''.join(c for c in keyword if c.isalnum() or c.isspace())
                                                if keyword:  # Only add non-empty keywords
                                                    positive_keywords.append(keyword)
                                    elif 'NEGATIVE KEYWORDS:' in section:
                                        # Extract keywords
                                        for line in section.split('\n'):
                                            if line.strip() and not line.startswith('NEGATIVE KEYWORDS:'):
                                                # Extract the keyword part before any parentheses
                                                keyword = line.strip('- ').strip().split('(')[0].strip()
                                                # Clean up the keyword (remove any remaining punctuation)
                                                keyword = ''.join(c for c in keyword if c.isalnum() or c.isspace())
                                                if keyword:  # Only add non-empty keywords
                                                    negative_keywords.append(keyword)
                            except Exception as e:
                                print(f"Error parsing LLM response: {e}")

                        # Count keyword frequencies
                        positive_freq = {}
                        negative_freq = {}
                        
                        # Convert transcript to lowercase for case-insensitive matching
                        transcript_lower = complete_transcript.lower()
                        
                        for keyword in positive_keywords:
                            # Convert keyword to lowercase for case-insensitive matching
                            keyword_lower = keyword.lower()
                            # Count occurrences using case-insensitive matching
                            count = transcript_lower.count(keyword_lower)
                            if count > 0:  # Only add keywords that appear at least once
                                positive_freq[keyword] = count
                        
                        for keyword in negative_keywords:
                            # Convert keyword to lowercase for case-insensitive matching
                            keyword_lower = keyword.lower()
                            # Count occurrences using case-insensitive matching
                            count = transcript_lower.count(keyword_lower)
                            if count > 0:  # Only add keywords that appear at least once
                                negative_freq[keyword] = count

                        # Save analysis results
                        analysis_result = {
                            'video_url': video_url,
                            'video_title': video_title,
                            'video_details': details,
                            'product_name': user_defined_product.strip(),
                            'overall_sentiment': overall_sentiment,
                            'positive_keywords': positive_keywords,
                            'negative_keywords': negative_keywords,
                            'positive_frequencies': positive_freq,
                            'negative_frequencies': negative_freq,
                            'summary': llm_analysis
                        }
                        
                        save_segment_product_analysis([analysis_result])
                        st.success("Analysis complete and saved! ✅")

                        # Display results
                        display_video_analysis(video_title, details, llm_analysis, positive_freq, negative_freq)

    # Product Review Dashboard
    st.sidebar.header("Filter by Product")
    all_analyzed_segments = load_segment_product_analysis()
    all_products = ["All Products"]
    if all_analyzed_segments:
        # Dynamically generate product list based on user input during analysis
        unique_products = set(seg.get('product_name', 'N/A') for seg in all_analyzed_segments)
        all_products.extend(sorted(list(unique_products)))

    selected_product = st.sidebar.selectbox("Select Product:", all_products)

    st.subheader("📊 Product Review Dashboard")

    filtered_segments = all_analyzed_segments
    if selected_product != "All Products" and all_analyzed_segments:
        filtered_segments = [seg for seg in all_analyzed_segments if seg.get('product_name', '').lower() == selected_product.lower()]

    if filtered_segments:
        st.subheader(f"Reviews for: {selected_product if selected_product != 'All Products' else 'All Products'}")
        
        # Combine all positive and negative keywords with their frequencies
        all_positive_freq = Counter()
        all_negative_freq = Counter()
        
        for segment in filtered_segments:
            # Add frequencies from each segment
            all_positive_freq.update(segment.get('positive_frequencies', {}))
            all_negative_freq.update(segment.get('negative_frequencies', {}))
        
        # Display top positive keywords with frequency > 1
        if all_positive_freq:
            st.subheader("Positive Keywords (Count > 1)")
            pos_data = {k: v for k, v in all_positive_freq.items() if v > 1}
            if pos_data:
                pos_df = pd.DataFrame({
                    'Keyword': list(pos_data.keys()),
                    'Total Count': list(pos_data.values())
                }).sort_values('Total Count', ascending=False)
                
                # Display bar chart
                st.bar_chart(pos_df.set_index('Keyword')['Total Count'])
                # Display table
                st.dataframe(pos_df)
            else:
                st.write("No positive keywords with count > 1")
        
        # Display top negative keywords with frequency > 1
        if all_negative_freq:
            st.subheader("Negative Keywords (Count > 1)")
            neg_data = {k: v for k, v in all_negative_freq.items() if v > 1}
            if neg_data:
                neg_df = pd.DataFrame({
                    'Keyword': list(neg_data.keys()),
                    'Total Count': list(neg_data.values())
                }).sort_values('Total Count', ascending=False)
                
                # Display bar chart
                st.bar_chart(neg_df.set_index('Keyword')['Total Count'])
                # Display table
                st.dataframe(neg_df)
            else:
                st.write("No negative keywords with count > 1")
        
        # Video Details Section
        st.subheader("📺 Video Details")
        
        # Create expandable sections for each video
        for segment in filtered_segments:
            with st.expander(f"📹 {segment.get('video_title', 'N/A')}"):
                # Video metadata
                video_details = segment.get('video_details', {})
                
                # Display basic stats in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    view_count = video_details.get('view_count')
                    st.write(f"**Views:** {view_count:,}" if isinstance(view_count, (int, float)) else f"**Views:** {view_count}")
                with col2:
                    like_count = video_details.get('like_count')
                    st.write(f"**Likes:** {like_count:,}" if isinstance(like_count, (int, float)) else f"**Likes:** {like_count}")
                with col3:
                    comment_count = video_details.get('comment_count')
                    st.write(f"**Comments:** {comment_count:,}" if isinstance(comment_count, (int, float)) else f"**Comments:** {comment_count}")
                
                # Display publish date and engagement metrics
                col1, col2 = st.columns(2)
                with col1:
                    publish_date = video_details.get('publish_date')
                    if publish_date:
                        from datetime import datetime
                        date_obj = datetime.strptime(publish_date, "%Y-%m-%dT%H:%M:%SZ")
                        formatted_date = date_obj.strftime("%B %d, %Y")
                        st.write(f"**Published:** {formatted_date}")
                with col2:
                    engagement_rate = video_details.get('engagement_rate')
                    if engagement_rate is not None:
                        st.write(f"**Engagement Rate:** {engagement_rate}%")
                
                # Display trending status
                if video_details.get('is_trending'):
                    st.success("🔥 This video is trending!")
                
                # Display summary
                if segment.get('summary'):
                    summary_section = segment['summary'].split('POSITIVE KEYWORDS:')[0].replace('SUMMARY:', '').strip()
                    st.write("**Summary:**")
                    st.write(summary_section)
                
                # Display positive keywords with count > 1
                pos_freq = segment.get('positive_frequencies', {})
                pos_data = {k: v for k, v in pos_freq.items() if v > 1}
                if pos_data:
                    st.write("**Positive Keywords:**")
                    pos_df = pd.DataFrame({
                        'Keyword': list(pos_data.keys()),
                        'Count': list(pos_data.values())
                    }).sort_values('Count', ascending=False)
                    st.dataframe(pos_df)
                
                # Display negative keywords with count > 1
                neg_freq = segment.get('negative_frequencies', {})
                neg_data = {k: v for k, v in neg_freq.items() if v > 1}
                if neg_data:
                    st.write("**Negative Keywords:**")
                    neg_df = pd.DataFrame({
                        'Keyword': list(neg_data.keys()),
                        'Count': list(neg_data.values())
                    }).sort_values('Count', ascending=False)
                    st.dataframe(neg_df)
    else:
        st.info("No product reviews analyzed yet.")

elif page == "Trend Analysis":
    show_trend_analysis() 