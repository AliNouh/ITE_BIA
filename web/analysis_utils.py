import streamlit as st
import numpy as np
from time import time
import psutil
import plotly.graph_objects as go
import plotly.express as px

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
    """Get system information"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.Process().memory_info().rss / (1024 * 1024)  # in megabytes
    return cpu_percent, memory

def create_performance_plots(results_df):
    """Create interactive performance plots"""
    # Compare accuracy and F1-score
    fig1 = go.Figure()
    for method in results_df['Method']:
        method_data = results_df[results_df['Method'] == method]
        fig1.add_trace(go.Bar(
            name=method,
            x=['Accuracy', 'F1 Score'],
            y=[method_data['Accuracy'].values[0], method_data['F1 Score'].values[0]],
            text=[f"{method_data['Accuracy'].values[0]:.3f}", 
                  f"{method_data['F1 Score'].values[0]:.3f}"],
            textposition='auto'
        ))
    
    fig1.update_layout(
        title="Performance Metrics Comparison",
        barmode='group',
        xaxis_title="Metric",
        yaxis_title="Value"
    )

    # Plot for selected features
    fig2 = px.bar(results_df,
                  x='Method',
                  y='Features',
                  title="Selected Features by Method",
                  text='Features',
                  color='Features',
                  color_continuous_scale='Viridis')
    
    return fig1, fig2

def display_detailed_results(results_df, original_features):
    """Display detailed analysis results"""
    # Best method
    best_method = results_df.loc[results_df['Accuracy'].idxmax()]
    
    # General statistics
    st.subheader("📊 Analysis Statistics")
    
    # Overall performance
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "🏆 Best Accuracy",
            f"{best_method['Accuracy']:.4f}",
            f"({best_method['Method']})"
        )
    with col2:
        avg_acc = results_df['Accuracy'].mean()
        st.metric(
            "📈 Average Accuracy",
            f"{avg_acc:.4f}"
        )
    with col3:
        feature_reduction = ((original_features - results_df['Features'].mean()) / original_features) * 100
        st.metric(
            "🎯 Average Feature Reduction",
            f"{feature_reduction:.1f}%"
        )

def update_progress(progress_bar, progress_text, status_text, 
                   progress_value, message, status=None):
    """Update progress state in the interface"""
    progress_bar.progress(progress_value)
    progress_text.markdown(f"### {message}")
    if status:
        status_text.info(status)