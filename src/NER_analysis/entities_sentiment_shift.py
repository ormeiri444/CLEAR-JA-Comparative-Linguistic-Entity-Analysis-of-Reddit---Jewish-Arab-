import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def analyze_sentiment_trends(csv_path, target_entities):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert timestamp to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'])

    # Convert entity_list strings to actual lists
    df['entity_list'] = df['entity_list'].fillna('[]')
    df['entity_list'] = df['entity_list'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    # Create a function to analyze trends for each entity
    def analyze_entity_trend(entity, community_df):
        # Filter posts containing the entity
        entity_posts = community_df[community_df['entity_list'].apply(lambda x: entity in x)]
        if entity_posts.empty:
            return None

        # Resample by day and calculate mean sentiment
        daily_sentiment = entity_posts.set_index('created_utc') \
            .resample('D')['sentiment_score'] \
            .mean() \
            .reset_index()

        # Calculate 14-day rolling average with centered window
        daily_sentiment['rolling_avg'] = daily_sentiment['sentiment_score'].rolling(
            window=30,
            center=True,
            min_periods=1
        ).mean()

        # Apply additional smoothing
        daily_sentiment['smooth_avg'] = daily_sentiment['rolling_avg'].rolling(
            window=15,
            center=True,
            min_periods=1
        ).mean()

        return daily_sentiment

    # Process each entity for both communities
    communities = ['arab', 'jewish']

    # Create subplots for each entity
    n_entities = len(target_entities)
    n_cols = 3
    n_rows = (n_entities + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 5 * n_rows))

    # Set style for smoother lines
    plt.style.use('seaborn')

    colors = {'arab': '#2ecc71', 'jewish': '#3498db'}

    for idx, entity in enumerate(target_entities, 1):
        ax = plt.subplot(n_rows, n_cols, idx)

        for community in communities:
            community_df = df[df['community'] == community]
            trend_data = analyze_entity_trend(entity, community_df)

            if trend_data is not None:
                # Plot the smoothed line
                plt.plot(trend_data['created_utc'],
                         trend_data['smooth_avg'],
                         label=f'{community.capitalize()}',
                         color=colors[community],
                         linewidth=2.5,
                         alpha=0.8)

        plt.title(f'Sentiment Trend: {entity}', pad=15)
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment')
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.xticks(rotation=45)

        # Add zero line for reference
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

        # Adjust y-axis limits for consistency
        plt.ylim(-1, 1)

    plt.tight_layout()
    plt.savefig('sentiment_trends.png', dpi=300, bbox_inches='tight')
    plt.close()


# Run the analysis with the specified entities
target_entities = ['Israel', 'Palestine', 'Hamas', 'USA', 'Iran', 'Trump',
                   'Biden', 'Netanyahu', 'IDF', 'UN', 'UNRWA', 'Hezbollah', 'Nasrallah']
analyze_sentiment_trends('/Users/ormeiri/Desktop/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-/data/combined_data_with_synthetic.csv', target_entities)