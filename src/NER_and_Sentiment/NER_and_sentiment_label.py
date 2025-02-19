import pandas as pd
import spacy
from transformers import pipeline
from datetime import datetime
import numpy as np
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import login

# Login to Hugging Face (if needed)
login(token="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")

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_trf")

# Initialize sentiment analyzer with a well-known model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def extract_entities_and_context(text, window_size=50):
    """
    Extract entities and their surrounding context from text.
    """
    if not isinstance(text, str):
        return []

    doc = nlp(text)
    entities_context = []

    for ent in doc.ents:
        # Get start and end indices for context window
        start = max(0, ent.start_char - window_size)
        end = min(len(text), ent.end_char + window_size)

        # Extract context
        context = text[start:end]

        entities_context.append({
            'entity': ent.text,
            'entity_type': ent.label_,
            'context': context
        })

    return entities_context


def analyze_sentiment(text):
    """
    Analyze sentiment of text using the sentiment analyzer.
    """
    if not isinstance(text, str) or not text.strip():
        return {'label': 'NEUTRAL', 'score': 0.5}

    try:
        result = sentiment_analyzer(text)[0]
        # Convert sentiment labels to match your expected format
        label = 'POSITIVE' if result['label'] == 'POSITIVE' else 'NEGATIVE'
        return {
            'label': label,
            'score': result['score']
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {'label': 'NEUTRAL', 'score': 0.5}


def process_data_for_analysis(df, source_type):
    """
    Process dataframe to extract entities and their sentiments.
    """
    results = []

    for _, row in df.iterrows():
        try:
            text = str(row['cleaned_text'])
            date = row['created_utc']

            # Extract entities and their context
            entities_context = extract_entities_and_context(text)

            # Analyze sentiment for each entity context
            for entity_info in entities_context:
                sentiment = analyze_sentiment(entity_info['context'])

                results.append({
                    'date': date if isinstance(date, datetime) else pd.to_datetime(date),
                    'entity': entity_info['entity'],
                    'entity_type': entity_info['entity_type'],
                    'context': entity_info['context'],
                    'sentiment_label': sentiment['label'],
                    'sentiment_score': sentiment['score'],
                    'source': source_type
                })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    return pd.DataFrame(results)


def analyze_temporal_sentiment(df):
    """
    Analyze sentiment changes over time for each entity.
    """
    # Add time-based features
    df['year_month'] = df['date'].dt.to_period('M')

    # Calculate average sentiment per entity per month
    temporal_analysis = df.groupby(['year_month', 'entity', 'source']).agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()

    temporal_analysis.columns = ['year_month', 'entity', 'source', 'avg_sentiment', 'std_sentiment', 'mention_count']

    return temporal_analysis


def compare_entity_sentiments(df):
    """
    Compare sentiment distributions between sources for each entity.
    """
    from scipy import stats

    comparison_results = {}

    for entity in df['entity'].unique():
        entity_data = df[df['entity'] == entity]

        if len(entity_data['source'].unique()) > 1:
            arab_sentiments = entity_data[entity_data['source'] == 'arab']['sentiment_score']
            jewish_sentiments = entity_data[entity_data['source'] == 'jewish']['sentiment_score']

            if len(arab_sentiments) > 0 and len(jewish_sentiments) > 0:
                # Perform Mann-Whitney U test
                statistic, pvalue = stats.mannwhitneyu(
                    arab_sentiments,
                    jewish_sentiments,
                    alternative='two-sided'
                )

                comparison_results[entity] = {
                    'statistic': statistic,
                    'p_value': pvalue,
                    'arab_mean': arab_sentiments.mean(),
                    'jewish_mean': jewish_sentiments.mean(),
                    'arab_count': len(arab_sentiments),
                    'jewish_count': len(jewish_sentiments)
                }

    return comparison_results


def visualize_temporal_trends(temporal_df, output_path):
    """
    Create visualizations for temporal sentiment trends.
    """
    # Convert year_month to datetime for plotting
    temporal_df['year_month'] = temporal_df['year_month'].astype(str)

    # Create line plot for top entities
    top_entities = temporal_df.groupby('entity')['mention_count'].sum().nlargest(10).index

    for entity in top_entities:
        entity_data = temporal_df[temporal_df['entity'] == entity]

        fig = px.line(
            entity_data,
            x='year_month',
            y='avg_sentiment',
            color='source',
            error_y='std_sentiment',
            title=f'Sentiment Trends for {entity}',
            labels={'year_month': 'Time', 'avg_sentiment': 'Average Sentiment'},
        )

        fig.write_html(f"{output_path}/sentiment_trend_{entity.replace(' ', '_')}.html")


def main():
    # Load cleaned data
    base_path = "/Users/ormeiri/Desktop/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-/data/processed"
    output_path = "/Users/ormeiri/Desktop/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-/data/cleaned_labeled_data"

    arab_data = []
    jewish_data = []

    # Process Arab subreddits
    for subreddit in ['arabs']:
        try:
            comments = pd.read_parquet(f"{base_path}/{subreddit}/processed/cleaned_comments.parquet")
            submissions = pd.read_parquet(f"{base_path}/{subreddit}/processed/processed_submissions.parquet")

            arab_data.extend([
                process_data_for_analysis(comments, 'arab'),
                process_data_for_analysis(submissions, 'arab')
            ])
        except Exception as e:
            print(f"Error processing {subreddit}: {e}")
            continue

    # Process Jewish subreddits
    for subreddit in ['Judaism']:
        try:
            comments = pd.read_parquet(f"{base_path}/{subreddit}/processed/cleaned_comments.parquet")
            submissions = pd.read_parquet(f"{base_path}/{subreddit}/processed/processed_submissions.parquet")

            jewish_data.extend([
                process_data_for_analysis(comments, 'jewish'),
                process_data_for_analysis(submissions, 'jewish')
            ])
        except Exception as e:
            print(f"Error processing {subreddit}: {e}")
            continue

    # Combine all data
    all_data = pd.concat(arab_data + jewish_data, ignore_index=True)

    # Perform temporal analysis
    temporal_results = analyze_temporal_sentiment(all_data)

    # Compare sentiments between sources
    comparison_results = compare_entity_sentiments(all_data)

    # Create visualizations
    visualize_temporal_trends(temporal_results, output_path)

    # Save results
    all_data.to_parquet(f"{output_path}/entity_sentiment_data.parquet")
    temporal_results.to_parquet(f"{output_path}/temporal_analysis.parquet")

    # Save comparison results
    comparison_df = pd.DataFrame.from_dict(comparison_results, orient='index')
    comparison_df.to_csv(f"{output_path}/sentiment_comparisons.csv")


if __name__ == "__main__":
    main()