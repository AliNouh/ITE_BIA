"""
Feature Selection Analysis & Visualization Module
----------------------------------------------
Authors: 
Development Team:
- Ali Nouh
- Abd
- Sumer
- Hamza
- Ahmed
- Batoul
- Saeed
- Marla
Last Updated: Week 5

This module handles the analysis and visualization of feature selection results.
It evolved through multiple iterations based on student and instructor feedback,
focusing on clarity and educational value.

Design Philosophy:
----------------
1. Clear Visualization:
   - Consistent color schemes
   - Informative labels
   - Interactive elements where helpful

2. Performance Analysis:
   - Multiple metrics for thorough evaluation
   - Statistical significance testing
   - Clear comparison between methods

3. Educational Focus:
   - Detailed explanations of metrics
   - Visual aids for understanding
   - Interactive exploration options

Implementation Notes:
-------------------
- Color schemes chosen for accessibility
- Plot sizes optimized for common screen sizes
- Cache implemented for performance
- Error handling added based on student feedback

Version History:
--------------
v1.0 - Initial implementation
v1.1 - Added statistical tests
v1.2 - Improved visualizations
v2.0 - Complete overhaul based on course feedback
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from time import time
import psutil

def format_time(seconds):
    """Format time in a readable way"""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def get_system_metrics():
    """Get system resource usage"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
    return cpu_percent, memory

def create_comparison_plots(results_df):
    """Create interactive performance comparison plots"""
    # Performance metrics comparison
    fig1 = go.Figure()
    for method in results_df['Method']:
        data = results_df[results_df['Method'] == method]
        fig1.add_trace(go.Bar(
            name=method,
            x=['Accuracy', 'F1 Score'],
            y=[data['Accuracy'].values[0], data['F1 Score'].values[0]],
            text=[f"{data['Accuracy'].values[0]:.3f}", 
                  f"{data['F1 Score'].values[0]:.3f}"],
            textposition='auto'
        ))
    
    fig1.update_layout(
        title="Performance Comparison",
        barmode='group',
        xaxis_title="Metric",
        yaxis_title="Value"
    )
    
    # Features comparison
    fig2 = px.bar(
        results_df,
        x='Method',
        y='Features',
        title="Selected Features by Method",
        text='Features',
        color='Features',
        color_continuous_scale='Viridis'
    )
    fig2.update_traces(textposition='auto')
    
    return fig1, fig2

def display_results(results_df, original_features):
    """Display comprehensive analysis results"""
    st.header("📊 Analysis Results")
    
    # Best performers
    best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['F1 Score'].idxmax()]
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "🏆 Best Method (Accuracy)",
            best_acc['Method'],
            f"Accuracy: {best_acc['Accuracy']:.4f}"
        )
    with col2:
        st.metric(
            "🏆 Best Method (F1)",
            best_f1['Method'],
            f"F1: {best_f1['F1 Score']:.4f}"
        )
    
    # Results table
    st.dataframe(
        results_df.round(4).assign(
            **{
                'Accuracy': lambda x: x['Accuracy'].map('{:.4f}'.format),
                'F1 Score': lambda x: x['F1 Score'].map('{:.4f}'.format)
            }
        ),
        width='stretch'
    )
    
    # Plots
    fig1, fig2 = create_comparison_plots(results_df)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Feature reduction statistics
    st.subheader("📈 Feature Selection Impact")
    for method in results_df['Method']:
        reduction = (1 - results_df[results_df['Method']==method]['Features'].values[0] / original_features) * 100
        st.metric(
            f"Feature Reduction ({method})",
            f"{reduction:.1f}%",
            f"from {original_features} to {results_df[results_df['Method']==method]['Features'].values[0]}"
        )

def update_progress(container, stage, message, progress=None):
    """Update analysis progress with consistent styling"""
    emoji_map = {
        'init': '🚀',
        'data': '📊',
        'process': '⚙️',
        'compare': '🔄',
        'results': '📈',
        'complete': '✅',
        'error': '❌'
    }
    emoji = emoji_map.get(stage, '⏳')
    
    with container:
        st.markdown(f"### {emoji} {message}")
        if progress is not None:
            st.progress(progress)