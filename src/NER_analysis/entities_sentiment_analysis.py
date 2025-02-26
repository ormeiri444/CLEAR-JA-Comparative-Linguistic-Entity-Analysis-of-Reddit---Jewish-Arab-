import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval


def analyze_entity_sentiments(csv_path, target_entities):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert entity_list strings to actual lists
    df['entity_list'] = df['entity_list'].fillna('[]')
    df['entity_list'] = df['entity_list'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    # Create separate dataframes for each community
    arab_df = df[df['community'] == 'arab']
    jewish_df = df[df['community'] == 'jewish']

    # Function to calculate average sentiment for entities
    def get_entity_sentiments(group_df, entities):
        results = []
        for entity in entities:
            # Filter posts containing the entity
            entity_posts = group_df[group_df['entity_list'].apply(lambda x: entity in x)]
            if not entity_posts.empty:
                avg_sentiment = entity_posts['sentiment_score'].mean()
                results.append({
                    'entity': entity,
                    'avg_sentiment': avg_sentiment,
                    'count': len(entity_posts)
                })
        return pd.DataFrame(results)

    # Get sentiments for both groups
    arab_sentiments = get_entity_sentiments(arab_df, target_entities)
    jewish_sentiments = get_entity_sentiments(jewish_df, target_entities)

    # Add community label
    arab_sentiments['community'] = 'Arab'
    jewish_sentiments['community'] = 'Jewish'

    # Combine results
    all_sentiments = pd.concat([arab_sentiments, jewish_sentiments])

    # Create the visualization
    plt.figure(figsize=(12, 6))

    # Create grouped bar plot
    sns.barplot(data=all_sentiments,
                x='entity',
                y='avg_sentiment',
                hue='community',
                palette=['#2ecc71', '#3498db'])

    plt.title('Average Sentiment by Entity and Community')
    plt.xlabel('Entity')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for i in plt.gca().containers:
        plt.gca().bar_label(i, fmt='%.2f')

    plt.tight_layout()
    plt.savefig('entity_sentiments.png')
    plt.close()

    # Create additional plot for post counts
    plt.figure(figsize=(12, 6))
    sns.barplot(data=all_sentiments,
                x='entity',
                y='count',
                hue='community',
                palette=['#2ecc71', '#3498db'])

    plt.title('Number of Posts by Entity and Community')
    plt.xlabel('Entity')
    plt.ylabel('Number of Posts')
    plt.xticks(rotation=45)

    # Add value labels
    for i in plt.gca().containers:
        plt.gca().bar_label(i)

    plt.tight_layout()
    plt.savefig('entity_post_counts.png')
    plt.close()


# Example usage:
target_entities = ['Israel', 'Palestine', 'Hamas', 'USA', 'Iran', 'Trump', 'Biden', 'Netanyahu', 'IDF', 'UN', 'UNRWA', 'Hezbollah', 'Nasrallah']
analyze_entity_sentiments('/Users/ormeiri/Desktop/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-/data/combined_data_with_synthetic.csv', target_entities)