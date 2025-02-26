import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "/Users/ormeiri/Desktop/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-/data/combined_data.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Count posts for Arab and Jewish communities
counts = df['community'].value_counts()[['arab', 'jewish']]

# Create a static bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(counts.index, counts.values, color=['blue', 'orange'], label=['Arab Posts', 'Jewish Posts'])

# Add labels and title
plt.xlabel("Community")
plt.ylabel("Number of Posts")
plt.title("Number of Posts by Community")

# Save as PNG
plt.savefig("posts_by_community.png", dpi=300, bbox_inches="tight")


# Group by community and sentiment
sentiment_counts = df.groupby(['community', 'sentiment_label']).size().unstack()
print(sentiment_counts)

# Plot
plt.figure(figsize=(6, 4))
sentiment_counts.loc[['arab', 'jewish']].plot(kind='bar', stacked=True, colormap='viridis', figsize=(6, 4))

# Labels and title
plt.xlabel("Community")
plt.ylabel("Number of Posts")
plt.title("Sentiment Distribution by Community")
plt.legend(title="Sentiment")

# Save as PNG
plt.savefig("sentiment_by_community.png", dpi=300, bbox_inches="tight")

# Show confirmation
print("Saved as sentiment_by_community.png")

# Convert created_utc to datetime
df['created_utc'] = pd.to_datetime(df['created_utc'])

# Sort by date
df = df.sort_values('created_utc')

# Create the plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Define window size for moving average (adjust as needed)
window_size = 5000

# Plot each community
for community in df['community'].unique():
    community_data = df[df['community'] == community].copy()

    # Calculate moving average
    community_data['smooth_sentiment'] = (
        community_data['sentiment_score']
        .rolling(window=window_size, min_periods=1, center=True)
        .mean()
    )

    # Plot the smoothed line
    plt.plot(
        community_data['created_utc'],
        community_data['smooth_sentiment'],
        label=community,
        linewidth=2.5
    )

    # Add confidence interval
    std = (
        community_data['sentiment_score']
        .rolling(window=window_size, min_periods=1, center=True)
        .std()
    )
    plt.fill_between(
        community_data['created_utc'],
        community_data['smooth_sentiment'] - std,
        community_data['smooth_sentiment'] + std,
        alpha=0.2
    )

# Customize the plot
plt.title('Smoothed Average Sentiment Over Time by Community', fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Sentiment Score', fontsize=12)
plt.legend(title='Community')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add horizontal line at neutral sentiment
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('smoothed_sentiment_trends.png', dpi=300, bbox_inches='tight')