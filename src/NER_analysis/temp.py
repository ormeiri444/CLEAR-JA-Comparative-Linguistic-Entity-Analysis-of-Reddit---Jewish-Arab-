import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

# Read the CSV file
df = pd.read_csv('/Users/ormeiri/Desktop/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-/data/combined_data.csv')


# Function to safely evaluate string representation of entity list
def safe_eval_entities(entity_str):
    if pd.isna(entity_str):
        return []
    try:
        return entity_str.split(', ')
    except:
        return []


# Create entity-sentiment pairs
entity_sentiments = []
for _, row in df.iterrows():
    entities = safe_eval_entities(row['entity_list'])
    for entity in entities:
        entity_sentiments.append({
            'community': row['community'],
            'entity': entity,
            'sentiment_score': row['sentiment_score']
        })

# Convert to DataFrame
entity_df = pd.DataFrame(entity_sentiments)

# Calculate average sentiment per entity per community
entity_stats = (entity_df.groupby(['community', 'entity'])
                .agg({
    'sentiment_score': ['mean', 'count', 'std']
})
                .reset_index())

entity_stats.columns = ['community', 'entity', 'avg_sentiment', 'mention_count', 'sentiment_std']

# Filter for entities mentioned 20 or more times
entity_stats = entity_stats[entity_stats['mention_count'] >= 2000]

# Print the number of entities that meet this criterion
print(f"Number of entities mentioned 20 or more times: {len(entity_stats)}")

# Create faceted plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Frequently Mentioned Entity Sentiment Analysis (≥20 mentions)', fontsize=16, y=1.02)

communities = ['jewish', 'arab']
for idx, community in enumerate(communities):
    community_data = entity_stats[entity_stats['community'] == community]

    if len(community_data) > 0:  # Only proceed if we have data for this community
        # Sort by sentiment for positive and negative
        top_positive = community_data.nlargest(10, 'avg_sentiment')
        top_negative = community_data.nsmallest(10, 'avg_sentiment')

        # Plot positive sentiments
        sns.barplot(
            data=top_positive,
            y='entity',
            x='avg_sentiment',
            ax=axes[idx, 0],
            color='green',
            alpha=0.6
        )
        axes[idx, 0].set_title(f'{community.capitalize()} Community - Most Positive Entities')
        axes[idx, 0].set_xlabel('Average Sentiment Score')

        # Add mention count to labels
        labels = [f'{row.entity}\n(n={int(row.mention_count)})' for _, row in top_positive.iterrows()]
        axes[idx, 0].set_yticklabels(labels)

        # Plot negative sentiments
        sns.barplot(
            data=top_negative,
            y='entity',
            x='avg_sentiment',
            ax=axes[idx, 1],
            color='red',
            alpha=0.6
        )
        axes[idx, 1].set_title(f'{community.capitalize()} Community - Most Negative Entities')
        axes[idx, 1].set_xlabel('Average Sentiment Score')

        # Add mention count to labels
        labels = [f'{row.entity}\n(n={int(row.mention_count)})' for _, row in top_negative.iterrows()]
        axes[idx, 1].set_yticklabels(labels)

        # Add error bars
        for ax in [axes[idx, 0], axes[idx, 1]]:
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
            ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('frequent_entity_sentiment_analysis.png', dpi=300, bbox_inches='tight')

# Print statistical summary
print("\nFrequently Mentioned Entity Sentiment Analysis Summary (≥20 mentions):")
for community in communities:
    community_data = entity_stats[entity_stats['community'] == community]
    if len(community_data) > 0:
        print(f"\n{community.capitalize()} Community:")
        print("\nMost Positive Entities:")
        print(community_data.nlargest(5, 'avg_sentiment')[['entity', 'avg_sentiment', 'mention_count']])
        print("\nMost Negative Entities:")
        print(community_data.nsmallest(5, 'avg_sentiment')[['entity', 'avg_sentiment', 'mention_count']])
    else:
        print(f"\n{community.capitalize()} Community: No entities with ≥20 mentions")

# Print total mentions for each entity across communities
print("\nTotal mentions for each entity across all communities:")
total_mentions = entity_df.groupby('entity')['sentiment_score'].count().sort_values(ascending=False)
print(total_mentions.head(10))