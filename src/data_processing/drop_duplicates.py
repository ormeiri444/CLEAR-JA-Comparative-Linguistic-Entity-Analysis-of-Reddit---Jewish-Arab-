import pandas as pd

def deduplicate_reddit_data(df, text_column='cleaned_text', keep='first', add_duplicate_count=True):
    """
    Remove duplicate entries from Reddit data based on text content.

    Args:
        df: DataFrame containing Reddit data
        text_column: Column to check for duplicates
        keep: Strategy for duplicates ('first', 'last', False)
        add_duplicate_count: Whether to add a column tracking duplicates

    Returns:
        Deduplicated DataFrame
    """
    # Remove duplicates
    deduplicated_df = df.drop_duplicates(subset=[text_column])

    return deduplicated_df

def process_parquet_file(file_path, text_column='cleaned_text', keep='first', add_duplicate_count=True):
    """
    Reads a Parquet file, deduplicates the data, and overwrites the same file.

    Args:
        file_path: Path to the input Parquet file.
        text_column: Column to check for duplicates.
        keep: Strategy for duplicates ('first', 'last', False).
        add_duplicate_count: Whether to add a column tracking duplicates.

    Returns:
        Deduplicated DataFrame.
    """
    # Load Parquet file
    df = pd.read_parquet(file_path)
    print(f"Length of original DataFrame: {len(df)}")

    # Deduplicate
    deduplicated_df = deduplicate_reddit_data(df, text_column, keep, add_duplicate_count)
    print(f"Length of deduplicated DataFrame: {len(deduplicated_df)}")

    # Overwrite the same Parquet file
    deduplicated_df.to_parquet(file_path, index=False)
    print(f"Deduplicated file overwritten at: {file_path}")

    return deduplicated_df

# Example usage:
process_parquet_file("/Users/ormeiri/Desktop/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-/data/processed/Jewish/processed/processed_submissions.parquet")

