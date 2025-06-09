import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def load_product_data():
    """Load and process product review data."""
    try:
        with open('data/product_reviews.json', 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return []

def aggregate_product_metrics(df):
    """Aggregate metrics for products with multiple videos."""
    # Group by product and calculate aggregated metrics
    agg_metrics = df.groupby('product_name').agg({
        'view_count': ['sum', 'mean', 'max'],
        'like_count': ['sum', 'mean', 'max'],
        'comment_count': ['sum', 'mean', 'max'],
        'engagement_rate': ['mean', 'max'],
        'publish_date': ['min', 'max', 'count']
    }).reset_index()
    
    # Flatten column names
    agg_metrics.columns = ['_'.join(col).strip('_') for col in agg_metrics.columns.values]
    
    # Rename columns for clarity
    agg_metrics = agg_metrics.rename(columns={
        'view_count_sum': 'total_views',
        'view_count_mean': 'avg_views',
        'view_count_max': 'max_views',
        'like_count_sum': 'total_likes',
        'like_count_mean': 'avg_likes',
        'like_count_max': 'max_likes',
        'comment_count_sum': 'total_comments',
        'comment_count_mean': 'avg_comments',
        'comment_count_max': 'max_comments',
        'engagement_rate_mean': 'avg_engagement_rate',
        'engagement_rate_max': 'max_engagement_rate',
        'publish_date_min': 'first_video_date',
        'publish_date_max': 'latest_video_date',
        'publish_date_count': 'video_count'
    })
    
    return agg_metrics

def track_video_changes(df):
    """Track changes in metrics for the same video over time."""
    # Group by video title and sort by analysis date
    video_changes = df.sort_values('analysis_date').groupby(['video_title', 'product_name'])
    
    changes_data = []
    for (video_title, product_name), group in video_changes:
        if len(group) > 1:  # Only process videos with multiple analyses
            # Calculate changes between consecutive analyses
            for i in range(1, len(group)):
                prev = group.iloc[i-1]
                curr = group.iloc[i]
                
                # Calculate absolute and percentage changes
                changes = {
                    'video_title': video_title,
                    'product_name': product_name,
                    'analysis_date': curr['analysis_date'],
                    'days_since_previous': (curr['analysis_date'] - prev['analysis_date']).days,
                    'views_change': curr['view_count'] - prev['view_count'],
                    'views_change_pct': ((curr['view_count'] - prev['view_count']) / prev['view_count'] * 100) if prev['view_count'] > 0 else 0,
                    'likes_change': curr['like_count'] - prev['like_count'],
                    'likes_change_pct': ((curr['like_count'] - prev['like_count']) / prev['like_count'] * 100) if prev['like_count'] > 0 else 0,
                    'comments_change': curr['comment_count'] - prev['comment_count'],
                    'comments_change_pct': ((curr['comment_count'] - prev['comment_count']) / prev['comment_count'] * 100) if prev['comment_count'] > 0 else 0,
                    'engagement_rate_change': curr['engagement_rate'] - prev['engagement_rate'],
                    'current_views': curr['view_count'],
                    'current_likes': curr['like_count'],
                    'current_comments': curr['comment_count'],
                    'current_engagement_rate': curr['engagement_rate']
                }
                changes_data.append(changes)
    
    return pd.DataFrame(changes_data) if changes_data else None

def analyze_video_trends(data):
    """Analyze video trends and create visualizations."""
    if not data:
        st.warning("No data available for analysis.")
        return

    # Convert data to DataFrame and add analysis date
    df = pd.DataFrame([
        {
            'video_title': item['video_title'],
            'view_count': item['video_details']['view_count'],
            'like_count': item['video_details']['like_count'],
            'comment_count': item['video_details']['comment_count'],
            'engagement_rate': item['video_details']['engagement_rate'],
            'publish_date': datetime.fromisoformat(item['video_details']['publish_date'].replace('Z', '+00:00')),
            'product_name': item['product_name'],
            'sentiment': item['overall_sentiment'],
            'analysis_date': datetime.now()  # Add current analysis date
        }
        for item in data
    ])

    # Sort by publish date
    df = df.sort_values('publish_date')

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Engagement Analysis", 
        "Sentiment Trends", 
        "Comment Analysis", 
        "Product Comparison",
        "Video Growth Analysis"
    ])

    with tab1:
        st.subheader("Video Engagement Analysis")
        
        # Engagement metrics over time
        fig = go.Figure()
        
        # Group by product to show metrics
        for product in df['product_name'].unique():
            product_data = df[df['product_name'] == product]
            
            # Add engagement rate line
            fig.add_trace(go.Scatter(
                x=product_data['publish_date'],
                y=product_data['engagement_rate'],
                name=f'{product} - Engagement Rate',
                mode='lines+markers',
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title='Engagement Rate Over Time by Product',
            xaxis_title='Date',
            yaxis_title='Engagement Rate (%)',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Engagement rate by product
        # Calculate engagement metrics
        product_engagement = df.groupby('product_name').agg({
            'engagement_rate': 'mean',  # Just using mean engagement rate
            'view_count': 'sum',
            'like_count': 'sum',
            'comment_count': 'sum',
            'video_title': 'count'
        }).reset_index()
        
        # Sort products by engagement rate for better visualization
        product_engagement = product_engagement.sort_values('engagement_rate', ascending=False)
        
        # Add detailed metrics table with improved formatting
        st.write("### Detailed Engagement Metrics by Product")
        detailed_metrics = product_engagement.rename(columns={
            'engagement_rate': 'Engagement Rate (%)',
            'view_count': 'Total Views',
            'like_count': 'Total Likes',
            'comment_count': 'Total Comments',
            'video_title': 'Number of Videos'
        })
        
        # Format the metrics table
        formatted_metrics = detailed_metrics.style.format({
            'Engagement Rate (%)': '{:.2f}%',
            'Total Views': '{:,.0f}',
            'Total Likes': '{:,.0f}',
            'Total Comments': '{:,.0f}',
            'Number of Videos': '{:,.0f}'
        }).background_gradient(subset=['Engagement Rate (%)'], cmap='YlOrRd')
        
        st.dataframe(formatted_metrics)

    with tab2:
        st.subheader("Sentiment Analysis")
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        fig3 = px.pie(values=sentiment_counts.values, 
                     names=sentiment_counts.index,
                     title='Overall Sentiment Distribution')
        st.plotly_chart(fig3)

        # Calculate sentiment counts for each product
        sentiment_by_product = df.groupby(['product_name', 'sentiment']).size().reset_index(name='count')
        
        # Add detailed sentiment metrics table
        st.write("### Detailed Sentiment Metrics by Product")
        sentiment_metrics = sentiment_by_product.pivot(
            index='product_name',
            columns='sentiment',
            values='count'
        ).fillna(0)
        
        # Calculate total videos for each product
        sentiment_metrics['Total Videos'] = sentiment_metrics.sum(axis=1)
        
        # Format the table
        formatted_sentiment = sentiment_metrics.style.format({
            'Total Videos': '{:.0f}'
        }).background_gradient(cmap='YlOrRd')
        
        st.dataframe(formatted_sentiment)

    with tab3:
        st.subheader("Comment Analysis")
        
        # Display comment statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Comments", df['comment_count'].sum())
        with col2:
            st.metric("Average Comments per Video", round(df['comment_count'].mean(), 2))
        with col3:
            st.metric("Max Comments", df['comment_count'].max())

        # Calculate detailed comment metrics by product
        comment_metrics = df.groupby('product_name').agg({
            'comment_count': ['sum', 'mean', 'max', 'count'],
            'view_count': 'sum',
            'like_count': 'sum'
        }).reset_index()
        
        # Flatten column names
        comment_metrics.columns = ['_'.join(col).strip('_') for col in comment_metrics.columns.values]
        
        # Rename columns for clarity
        comment_metrics = comment_metrics.rename(columns={
            'comment_count_sum': 'Total Comments',
            'comment_count_mean': 'Avg Comments per Video',
            'comment_count_max': 'Max Comments',
            'comment_count_count': 'Number of Videos',
            'view_count_sum': 'Total Views',
            'like_count_sum': 'Total Likes'
        })
        
        # Calculate comment engagement rate
        comment_metrics['Comment Engagement Rate'] = (comment_metrics['Total Comments'] / comment_metrics['Total Views'] * 100).round(2)
        
        # Sort by total comments
        comment_metrics = comment_metrics.sort_values('Total Comments', ascending=False)
        
        # Format the table
        formatted_comment_metrics = comment_metrics.style.format({
            'Total Comments': '{:,.0f}',
            'Avg Comments per Video': '{:.2f}',
            'Max Comments': '{:,.0f}',
            'Number of Videos': '{:,.0f}',
            'Total Views': '{:,.0f}',
            'Total Likes': '{:,.0f}',
            'Comment Engagement Rate': '{:.2f}%'
        }).background_gradient(subset=['Comment Engagement Rate'], cmap='YlOrRd')
        
        st.write("### Detailed Comment Metrics by Product")
        st.dataframe(formatted_comment_metrics)

    with tab4:
        st.subheader("Product Comparison")
        
        # Create comparison options
        st.write("### Video Comparison")
        
        # Get unique products
        products = df['product_name'].unique()
        
        # Create two columns for selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Select First Video")
            first_product = st.selectbox("Product 1", ['Select Product'] + list(products), key="first_product")
            if first_product != 'Select Product':
                first_videos = df[df['product_name'] == first_product]['video_title'].unique()
                first_video = st.selectbox("Video 1", ['Select Video'] + list(first_videos), key="first_video")
            else:
                first_video = 'Select Video'
            
        with col2:
            st.write("Select Second Video")
            second_product = st.selectbox("Product 2", ['Select Product'] + list(products), key="second_product")
            if second_product != 'Select Product':
                second_videos = df[df['product_name'] == second_product]['video_title'].unique()
                second_video = st.selectbox("Video 2", ['Select Video'] + list(second_videos), key="second_video")
            else:
                second_video = 'Select Video'
        
        # Only show comparison if both videos are selected
        if first_video != 'Select Video' and second_video != 'Select Video':
            # Get selected video data
            first_video_data = df[df['video_title'] == first_video].iloc[0]
            second_video_data = df[df['video_title'] == second_video].iloc[0]
            
            # Create comparison metrics
            metrics = [
                ('Product Name', first_video_data['product_name'], second_video_data['product_name']),
                ('Video Title', first_video_data['video_title'], second_video_data['video_title']),
                ('Views', f"{first_video_data['view_count']:,}", f"{second_video_data['view_count']:,}"),
                ('Likes', f"{first_video_data['like_count']:,}", f"{second_video_data['like_count']:,}"),
                ('Comments', f"{first_video_data['comment_count']:,}", f"{second_video_data['comment_count']:,}"),
                ('Engagement Rate', f"{first_video_data['engagement_rate']:.2f}%", f"{second_video_data['engagement_rate']:.2f}%"),
                ('Publish Date', first_video_data['publish_date'].strftime('%Y-%m-%d'), second_video_data['publish_date'].strftime('%Y-%m-%d')),
                ('Sentiment', first_video_data['sentiment'], second_video_data['sentiment'])
            ]
            
            # Calculate differences
            differences = []
            for metric, val1, val2 in metrics:
                if '%' in str(val1):
                    diff = f"{float(val1.replace('%', '').replace(',', '')) - float(val2.replace('%', '').replace(',', '')):.2f}%"
                elif str(val1).replace(',', '').replace('.', '').isdigit():
                    diff = f"{int(val1.replace(',', '')) - int(val2.replace(',', '')):,}"
                else:
                    diff = 'N/A'
                differences.append(diff)
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Metric': [m[0] for m in metrics],
                'Video 1': [m[1] for m in metrics],
                'Video 2': [m[2] for m in metrics],
                'Difference': differences
            })
            
            # Format the comparison table
            st.write("### Video Comparison Results")
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("Please select both videos to see the comparison.")

    with tab5:
        st.subheader("Video Growth Analysis")
        
        # Track changes in metrics over time
        changes_df = track_video_changes(df)
        
        if changes_df is not None and not changes_df.empty:
            # Filter for recent changes (last 30 days)
            recent_changes = changes_df[changes_df['days_since_previous'] <= 30]
            
            if not recent_changes.empty:
                # Add video selection
                st.write("### Select Video for Analysis")
                
                # Get unique products
                products = recent_changes['product_name'].unique()
                
                # Product selection
                selected_product = st.selectbox(
                    "Select Product",
                    ['All Products'] + list(products),
                    key="growth_product"
                )
                
                # Filter videos based on selected product
                if selected_product != 'All Products':
                    filtered_changes = recent_changes[recent_changes['product_name'] == selected_product]
                    videos = filtered_changes['video_title'].unique()
                else:
                    filtered_changes = recent_changes
                    videos = filtered_changes['video_title'].unique()
                
                # Video selection
                selected_videos = st.multiselect(
                    "Select Videos to Compare",
                    videos,
                    default=videos[:2] if len(videos) > 1 else videos[:1],
                    key="growth_videos"
                )
                
                if selected_videos:
                    # Filter data for selected videos
                    filtered_changes = filtered_changes[filtered_changes['video_title'].isin(selected_videos)]
                    
                    st.write("### Time Series Trends")
                    
                    # Views trend
                    st.write("#### Views Trend Over Time")
                    max_views = filtered_changes['current_views'].max()
                    fig_views = go.Figure()
                    
                    # Add data points
                    for video in selected_videos:
                        video_data = filtered_changes[
                            (filtered_changes['video_title'] == video) & 
                            (filtered_changes['product_name'] == selected_product)
                        ]
                        if not video_data.empty:
                            fig_views.add_trace(go.Scatter(
                                x=video_data['analysis_date'],
                                y=video_data['current_views'],
                                name=f"{video} ({selected_product})",
                                mode='lines+markers',
                                marker=dict(size=8)
                            ))
                    
                    fig_views.update_layout(
                        showlegend=True,
                        height=400,
                        yaxis=dict(
                            title="Views",
                            tickformat=",d",
                            range=[0, max_views * 1.1]
                        ),
                        xaxis=dict(
                            title="Date"
                        )
                    )
                    st.plotly_chart(fig_views, use_container_width=True)
                    
                    # Likes trend
                    st.write("#### Likes Trend Over Time")
                    max_likes = filtered_changes['current_likes'].max()
                    fig_likes = go.Figure()
                    
                    # Add data points
                    for video in selected_videos:
                        video_data = filtered_changes[
                            (filtered_changes['video_title'] == video) & 
                            (filtered_changes['product_name'] == selected_product)
                        ]
                        if not video_data.empty:
                            fig_likes.add_trace(go.Scatter(
                                x=video_data['analysis_date'],
                                y=video_data['current_likes'],
                                name=f"{video} ({selected_product})",
                                mode='lines+markers',
                                marker=dict(size=8)
                            ))
                    
                    fig_likes.update_layout(
                        showlegend=True,
                        height=400,
                        yaxis=dict(
                            title="Likes",
                            tickformat=",d",
                            range=[0, max_likes * 1.1]
                        ),
                        xaxis=dict(
                            title="Date"
                        )
                    )
                    st.plotly_chart(fig_likes, use_container_width=True)
                    
                    # Comments trend
                    st.write("#### Comments Trend Over Time")
                    max_comments = filtered_changes['current_comments'].max()
                    fig_comments = go.Figure()
                    
                    # Add data points
                    for video in selected_videos:
                        video_data = filtered_changes[
                            (filtered_changes['video_title'] == video) & 
                            (filtered_changes['product_name'] == selected_product)
                        ]
                        if not video_data.empty:
                            fig_comments.add_trace(go.Scatter(
                                x=video_data['analysis_date'],
                                y=video_data['current_comments'],
                                name=f"{video} ({selected_product})",
                                mode='lines+markers',
                                marker=dict(size=8)
                            ))
                    
                    fig_comments.update_layout(
                        showlegend=True,
                        height=400,
                        yaxis=dict(
                            title="Comments",
                            tickformat=",d",
                            range=[0, max_comments * 1.1]
                        ),
                        xaxis=dict(
                            title="Date"
                        )
                    )
                    st.plotly_chart(fig_comments, use_container_width=True)
                    
                    # Display the raw data
                    st.write("### Raw Growth Data")
                    st.dataframe(filtered_changes[[
                        'video_title',
                        'product_name',
                        'analysis_date',
                        'current_views',
                        'current_likes',
                        'current_comments',
                        'days_since_previous'
                    ]].sort_values('analysis_date', ascending=False))
                else:
                    st.info("Please select at least one video to view the analysis.")
            else:
                st.info("No recent changes detected in the last 30 days.")
        else:
            st.info("No video metric changes detected. Try analyzing the same video again after some time.")

def show_trend_analysis():
    """Main function to display the trend analysis page."""
    st.title("Video Trend Analysis")
    
    # Load data
    data = load_product_data()
    
    if data:
        # Add filters
        st.sidebar.header("Filters")
        
        # Product filter
        products = list(set(item['product_name'] for item in data))
        selected_products = st.sidebar.multiselect(
            "Select Products",
            products,
            default=products
        )
        
        # Date range filter
        dates = [datetime.fromisoformat(item['video_details']['publish_date'].replace('Z', '+00:00')) 
                for item in data]
        min_date = min(dates)
        max_date = max(dates)
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data based on selections
        filtered_data = [
            item for item in data
            if item['product_name'] in selected_products
            and min_date <= datetime.fromisoformat(item['video_details']['publish_date'].replace('Z', '+00:00')) <= max_date
        ]
        
        # Analyze trends
        analyze_video_trends(filtered_data)
        
        # Display raw data
        if st.checkbox("Show Raw Data"):
            st.dataframe(pd.DataFrame([
                {
                    'Video Title': item['video_title'],
                    'Product': item['product_name'],
                    'Views': item['video_details']['view_count'],
                    'Likes': item['video_details']['like_count'],
                    'Comments': item['video_details']['comment_count'],
                    'Engagement Rate': f"{item['video_details']['engagement_rate']:.2f}%",
                    'Publish Date': item['video_details']['publish_date'],
                    'Sentiment': item['overall_sentiment']
                }
                for item in filtered_data
            ]))
    else:
        st.error("No data available for analysis.") 