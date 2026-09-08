import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

def plot_sentiment_distribution(df):
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    color_map = {'Positive': '#22c55e', 'Neutral': '#94a3b8', 'Negative': '#ef4444'}
    
    fig = px.pie(sentiment_counts, values='Count', names='Sentiment', hole=0.6,
                 color='Sentiment', color_discrete_map=color_map,
                 title='Sentiment Distribution')
    
    fig.update_layout(
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="sans serif", color="#0f172a")
    )
    return fig

def plot_sentiment_trend(df):
    if 'date' not in df.columns:
        return None
        
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    if len(df) == 0:
        return None
    
    daily_sentiment = df.groupby([df['date'].dt.date, 'sentiment']).size().reset_index(name='count')
    
    color_map = {'Positive': '#22c55e', 'Neutral': '#94a3b8', 'Negative': '#ef4444'}
    
    fig = px.bar(daily_sentiment, x='date', y='count', color='sentiment',
                  color_discrete_map=color_map, barmode='stack',
                  labels={'date': 'Date', 'count': 'Number of Articles'},
                  title='Sentiment Over Time')
                  
    fig.update_layout(
        margin=dict(t=40, b=10, l=10, r=10), hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="sans serif", color="#0f172a")
    )
    return fig

def plot_source_sentiment(df):
    if 'country' not in df.columns:
        return None
        
    source_sentiment = df.groupby(['country', 'sentiment']).size().reset_index(name='count')
    top_sources = df['country'].value_counts().nlargest(5).index
    source_sentiment = source_sentiment[source_sentiment['country'].isin(top_sources)]
    
    color_map = {'Positive': '#22c55e', 'Neutral': '#94a3b8', 'Negative': '#ef4444'}
    
    fig = px.bar(source_sentiment, x='country', y='count', color='sentiment', barmode='group',
                 color_discrete_map=color_map,
                 labels={'country': 'Source', 'count': 'Articles'},
                 title='Sentiment by Top Sources')
                 
    fig.update_layout(
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="sans serif", color="#0f172a")
    )
    return fig

def generate_wordcloud(text_series, title="Most Common Words"):
    text = " ".join(text_series.dropna().astype(str))
    
    if not text.strip():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No text available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
        
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='cividis', max_words=100, contour_width=0).generate(text)
                          
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, color="#0f172a", pad=20)
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    return fig

def plot_statistical_bar_chart(df):
    """
    Generates a bar chart showing article count for Positive, Negative, and Neutral categories.
    """
    categories = ['Positive', 'Negative', 'Neutral']
    counts = [int((df['sentiment'] == cat).sum()) for cat in categories]
    
    stat_df = pd.DataFrame({
        'Sentiment Category': categories,
        'No. of Articles': counts
    })
    
    color_map = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#94a3b8'}
    
    fig = px.bar(
        stat_df,
        x='Sentiment Category',
        y='No. of Articles',
        color='Sentiment Category',
        color_discrete_map=color_map,
        text='No. of Articles',
        title='Sentiment Distribution (Dataset)'
    )
    
    fig.update_traces(
        textposition='outside',
        textfont_size=13,
        textfont_color='#0f172a'
    )
    
    fig.update_layout(
        xaxis_title="Sentiment Category",
        yaxis_title="No. of Articles",
        showlegend=False,
        margin=dict(t=50, b=40, l=40, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="sans serif", color="#0f172a", size=12),
        height=380
    )
    return fig

def plot_statistical_donut_chart(df):
    """
    Generates a donut chart showing sentiment percentage breakdown for the dataset.
    """
    categories = ['Positive', 'Negative', 'Neutral']
    counts = [int((df['sentiment'] == cat).sum()) for cat in categories]
    total = sum(counts) if sum(counts) > 0 else 1
    
    pcts = [round((c / total) * 100, 1) for c in counts]
    labels_with_pct = [f"{cat} ({p}%)" for cat, p in zip(categories, pcts)]
    
    stat_df = pd.DataFrame({
        'Sentiment': categories,
        'Count': counts,
        'Percentage': pcts,
        'LegendLabel': labels_with_pct
    })
    
    color_map = {
        'Positive': '#22c55e',
        'Negative': '#ef4444',
        'Neutral': '#94a3b8'
    }
    
    fig = px.pie(
        stat_df,
        values='Count',
        names='LegendLabel',
        color='Sentiment',
        color_discrete_map=color_map,
        hole=0.55,
        title='Sentiment Percentage'
    )
    
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate='%{label}<br>Count: %{value}<extra></extra>'
    )
    
    fig.update_layout(
        margin=dict(t=50, b=40, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="sans serif", color="#0f172a", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=380
    )
    return fig

