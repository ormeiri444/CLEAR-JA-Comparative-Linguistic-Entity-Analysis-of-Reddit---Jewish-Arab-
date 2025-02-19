import zstandard as zstd
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm

JEWISH_SUBREDDITS = {
}

ARAB_SUBREDDITS = {
 'Palestine',
}

# Dictionary to map subreddits to their communities
SUBREDDIT_TO_COMMUNITY = {
                             subreddit: 'jewish' for subreddit in JEWISH_SUBREDDITS
                         } | {
                             subreddit: 'arab' for subreddit in ARAB_SUBREDDITS
                         }

# ALL_SUBREDDITS = JEWISH_SUBREDDITS.union(ARAB_SUBREDDITS)
ALL_SUBREDDITS = ARAB_SUBREDDITS

def read_zst_file(file_path):
    """Read compressed .zst file line by line with robust UTF-8 handling."""
    buffer = ""
    with open(file_path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            while True:
                chunk = reader.read(2 ** 20)  # 1mb chunks (smaller for better handling)
                if not chunk:
                    break

                try:
                    buffer += chunk.decode('utf-8', errors='replace')
                except UnicodeDecodeError as e:
                    # If decode fails, try to find a safe breaking point
                    partial = chunk[:e.start].decode('utf-8', errors='replace')
                    buffer += partial

                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():  # Skip empty lines
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

            # Process any remaining data in buffer
            if buffer.strip():
                try:
                    yield json.loads(buffer)
                except json.JSONDecodeError:
                    pass


def process_file(file_path, output_dir, file_type):
    """Process a single Reddit data file and save relevant subreddit data."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if this file has been fully processed by looking for all expected output files
    date_str = Path(file_path).stem.split('_')[1]  # Extract date from filename
    # all_processed = True
    # for subreddit in ALL_SUBREDDITS:
    #     expected_output = os.path.join(output_dir, subreddit, f"{file_type}_{date_str}.parquet")
    #     if not os.path.exists(expected_output):
    #         all_processed = False
    #         break
    #
    # if all_processed:
    #     print(f"Skipping {file_path} - already processed")
    #     return

    # Initialize DataFrames for each subreddit
    subreddit_data = {subreddit: [] for subreddit in ALL_SUBREDDITS}

    # Process the file
    for item in tqdm(read_zst_file(file_path), desc=f"Processing {file_path}"):
        subreddit = item.get('subreddit')

        if subreddit in ALL_SUBREDDITS:
            # Extract relevant fields
            processed_item = {
                'id': item.get('id'),
                'created_utc': datetime.fromtimestamp(item.get('created_utc', 0)),
                'subreddit': subreddit,
                'community': SUBREDDIT_TO_COMMUNITY[subreddit],
                'author': item.get('author'),
                'text': item.get('selftext' if file_type == 'submission' else 'body', ''),
                'title': item.get('title', '') if file_type == 'submission' else '',
                'score': item.get('score', 0),
                'parent_id': str(item.get('parent_id', '') if file_type == 'comment' else ''),
                'type': file_type
            }

            subreddit_data[subreddit].append(processed_item)

    # Save data for each subreddit
    for subreddit, data in subreddit_data.items():
        if data:  # Only save if we have data
            df = pd.DataFrame(data)

            # Create subreddit-specific directory
            subreddit_dir = os.path.join(output_dir, subreddit)
            os.makedirs(subreddit_dir, exist_ok=True)

            # Save to parquet with date in filename
            date_str = Path(file_path).stem.split('_')[1]  # Extract date from filename
            output_path = os.path.join(subreddit_dir, f"{file_type}_{date_str}.parquet")
            df.to_parquet(output_path, index=False)

            print(f"Saved {len(data)} {file_type}s for r/{subreddit} to {output_path}")


def process_reddit_data(base_dir, output_dir):
    """Process all Reddit data files in the given directory."""
    print("\nChecking for existing processed files...")
    # Process submissions
    submissions_dir = os.path.join(base_dir, 'submissions')
    if os.path.exists(submissions_dir):
        for file in os.listdir(submissions_dir):
            if file.endswith('.zst'):
                # Check if this file has been fully processed by looking for all expected output files
                date_str = Path(file).stem.split('_')[1]  # Extract date from filename
                all_processed = True
                for subreddit in ALL_SUBREDDITS:
                    expected_output = os.path.join(output_dir, subreddit, f"submission_{date_str}.parquet")
                    if not os.path.exists(expected_output):
                        all_processed = False
                        break
                if all_processed:
                    print(f"Skipping {file} - already processed")
                    continue
                file_path = os.path.join(submissions_dir, file)
                process_file(file_path, output_dir, 'submission')

    # Process comments
    comments_dir = os.path.join(base_dir, 'comments')
    if os.path.exists(comments_dir):
        for file in os.listdir(comments_dir):
            if file.endswith('.zst'):
                # Check if this file has been fully processed by looking for all expected output files
                date_str = Path(file).stem.split('_')[1]  # Extract date from filename
                all_processed = True
                for subreddit in ALL_SUBREDDITS:
                    expected_output = os.path.join(output_dir, subreddit, f"comment_{date_str}.parquet")
                    if not os.path.exists(expected_output):
                        all_processed = False
                        break
                if all_processed:
                    print(f"Skipping {file} - already processed")
                    continue
                file_path = os.path.join(comments_dir, file)
                process_file(file_path, output_dir, 'comment')

if __name__ == "__main__":
    # Configuration
    BASE_DIR = "/Users/ormeiri/Downloads/reddit"  # Directory containing submissions and comments folders
    OUTPUT_DIR = "/Users/ormeiri/Desktop/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-/data/processed"  # Directory where processed data will be saved

    # Process all data
    process_reddit_data(BASE_DIR, OUTPUT_DIR)