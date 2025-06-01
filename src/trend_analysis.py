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
    video_changes = df.sort_values('analysis_date').groupby('video_title')
    
    changes_data = []
    for video_title, group in video_changes:
        if len(group) > 1:  # Only process videos with multiple analyses
            # Calculate changes between consecutive analyses
            for i in range(1, len(group)):
                prev = group.iloc[i-1]
                curr = group.iloc[i]
                
                # Calculate absolute and percentage changes
                changes = {
                    'video_title': video_title,
                    'product_name': curr['product_name'],
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
        fig.add_trace(go.Scatter(x=df['publish_date'], y=df['view_count'], 
                                name='Views', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=df['publish_date'], y=df['like_count'], 
                                name='Likes', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=df['publish_date'], y=df['comment_count'], 
                                name='Comments', mode='lines+markers'))
        
        fig.update_layout(title='Engagement Metrics Over Time',
                         xaxis_title='Date',
                         yaxis_title='Count',
                         hovermode='x unified')
        st.plotly_chart(fig)

        # Engagement rate by product
        fig2 = px.bar(df, x='product_name', y='engagement_rate',
                     title='Engagement Rate by Product',
                     labels={'engagement_rate': 'Engagement Rate (%)', 'product_name': 'Product'})
        st.plotly_chart(fig2)

    with tab2:
        st.subheader("Sentiment Analysis")
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        fig3 = px.pie(values=sentiment_counts.values, 
                     names=sentiment_counts.index,
                     title='Overall Sentiment Distribution')
        st.plotly_chart(fig3)

        # Sentiment by product
        fig4 = px.bar(df, x='product_name', color='sentiment',
                     title='Sentiment Distribution by Product',
                     barmode='group')
        st.plotly_chart(fig4)

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

        # Comment engagement ratio
        df['comment_engagement_ratio'] = df['comment_count'] / df['view_count'] * 100
        fig5 = px.bar(df, x='video_title', y='comment_engagement_ratio',
                     title='Comment Engagement Ratio by Video',
                     labels={'comment_engagement_ratio': 'Comment Engagement Ratio (%)', 
                            'video_title': 'Video Title'})
        fig5.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig5)

    with tab4:
        st.subheader("Product Comparison")
        
        # Aggregate metrics for products with multiple videos
        agg_metrics = aggregate_product_metrics(df)
        
        # Display product comparison metrics
        st.write("### Product Performance Metrics")
        st.dataframe(agg_metrics)
        
        # Create comparison visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Total engagement metrics by product
            fig6 = go.Figure()
            fig6.add_trace(go.Bar(name='Total Views', x=agg_metrics['product_name'], y=agg_metrics['total_views']))
            fig6.add_trace(go.Bar(name='Total Likes', x=agg_metrics['product_name'], y=agg_metrics['total_likes']))
            fig6.add_trace(go.Bar(name='Total Comments', x=agg_metrics['product_name'], y=agg_metrics['total_comments']))
            
            fig6.update_layout(title='Total Engagement by Product',
                             barmode='group',
                             xaxis_title='Product',
                             yaxis_title='Count')
            st.plotly_chart(fig6)
        
        with col2:
            # Average engagement metrics by product
            fig7 = go.Figure()
            fig7.add_trace(go.Bar(name='Avg Views', x=agg_metrics['product_name'], y=agg_metrics['avg_views']))
            fig7.add_trace(go.Bar(name='Avg Likes', x=agg_metrics['product_name'], y=agg_metrics['avg_likes']))
            fig7.add_trace(go.Bar(name='Avg Comments', x=agg_metrics['product_name'], y=agg_metrics['avg_comments']))
            
            fig7.update_layout(title='Average Engagement per Video by Product',
                             barmode='group',
                             xaxis_title='Product',
                             yaxis_title='Average Count')
            st.plotly_chart(fig7)
        
        # Video count and engagement rate
        fig8 = px.scatter(agg_metrics, 
                         x='video_count', 
                         y='avg_engagement_rate',
                         size='total_views',
                         color='product_name',
                         title='Video Count vs Engagement Rate',
                         labels={'video_count': 'Number of Videos',
                                'avg_engagement_rate': 'Average Engagement Rate (%)',
                                'total_views': 'Total Views'})
        st.plotly_chart(fig8)

    with tab5:
        st.subheader("Video Growth Analysis")
        
        # Track changes in metrics over time
        changes_df = track_video_changes(df)
        
        if changes_df is not None and not changes_df.empty:
            # Filter for recent changes (last 30 days)
            recent_changes = changes_df[changes_df['days_since_previous'] <= 30]
            
            if not recent_changes.empty:
                st.write("### Recent Metric Changes (Last 30 Days)")
                
                # Display changes in a table
                st.dataframe(recent_changes[[
                    'video_title', 'product_name', 'days_since_previous',
                    'views_change', 'views_change_pct',
                    'likes_change', 'likes_change_pct',
                    'comments_change', 'comments_change_pct',
                    'engagement_rate_change'
                ]].round(2))
                
                # Create visualizations for metric changes
                col1, col2 = st.columns(2)
                
                with col1:
                    # Views growth over time
                    fig9 = px.bar(recent_changes,
                                x='video_title',
                                y='views_change_pct',
                                title='Views Growth Rate (%)',
                                labels={'views_change_pct': 'Views Growth (%)',
                                       'video_title': 'Video Title'})
                    fig9.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig9)
                
                with col2:
                    # Engagement rate changes
                    fig10 = px.bar(recent_changes,
                                 x='video_title',
                                 y='engagement_rate_change',
                                 title='Engagement Rate Change',
                                 labels={'engagement_rate_change': 'Engagement Rate Change (%)',
                                        'video_title': 'Video Title'})
                    fig10.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig10)
                
                # Growth trends
                st.write("### Growth Trends")
                
                # Calculate average daily growth rates
                recent_changes['daily_views_growth'] = recent_changes['views_change'] / recent_changes['days_since_previous']
                recent_changes['daily_likes_growth'] = recent_changes['likes_change'] / recent_changes['days_since_previous']
                recent_changes['daily_comments_growth'] = recent_changes['comments_change'] / recent_changes['days_since_previous']
                
                # Plot daily growth rates
                fig11 = go.Figure()
                fig11.add_trace(go.Bar(name='Views/Day', x=recent_changes['video_title'], 
                                     y=recent_changes['daily_views_growth']))
                fig11.add_trace(go.Bar(name='Likes/Day', x=recent_changes['video_title'], 
                                     y=recent_changes['daily_likes_growth']))
                fig11.add_trace(go.Bar(name='Comments/Day', x=recent_changes['video_title'], 
                                     y=recent_changes['daily_comments_growth']))
                
                fig11.update_layout(title='Average Daily Growth Rates',
                                  barmode='group',
                                  xaxis_title='Video Title',
                                  yaxis_title='Average Daily Growth',
                                  xaxis_tickangle=-45)
                st.plotly_chart(fig11)
                
                # Projected growth
                st.write("### Projected Growth (Next 7 Days)")
                
                # Calculate projected metrics
                projections = recent_changes.copy()
                projections['projected_views'] = projections['current_views'] + (projections['daily_views_growth'] * 7)
                projections['projected_likes'] = projections['current_likes'] + (projections['daily_likes_growth'] * 7)
                projections['projected_comments'] = projections['current_comments'] + (projections['daily_comments_growth'] * 7)
                
                # Display projections
                st.dataframe(projections[[
                    'video_title', 'current_views', 'projected_views',
                    'current_likes', 'projected_likes',
                    'current_comments', 'projected_comments'
                ]].round(0))
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