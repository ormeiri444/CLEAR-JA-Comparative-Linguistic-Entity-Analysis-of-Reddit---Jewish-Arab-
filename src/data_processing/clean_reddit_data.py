import pandas as pd
import os
from pathlib import Path
from transformers import pipeline
from tqdm.auto import tqdm
import time

# Initialize language detection model globally
lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")


def detect_language(text):
    """
    Detect the language of the text and return language and confidence.
    Handles long texts by splitting into chunks and using majority voting.
    """
    try:
        # Split text into sentences (rough approximation)
        chunks = text.replace('!', '.').replace('?', '.').split('.')
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]

        if not chunks:
            chunks = [text]

        # Process each chunk
        results = []
        for chunk in chunks[:5]:  # Process up to 5 chunks for efficiency
            try:
                result = lang_detector(chunk[:512])[0]  # Limit chunk size to 512 characters
                results.append((result['label'], result['score']))
            except Exception:
                continue

        if not results:
            return 'en', 0.0

        # Count language occurrences and average confidence by language
        lang_scores = {}
        for lang, score in results:
            if lang not in lang_scores:
                lang_scores[lang] = {'count': 0, 'total_score': 0.0}
            lang_scores[lang]['count'] += 1
            lang_scores[lang]['total_score'] += score

        # Find the most common language
        max_count = 0
        max_lang = 'en'
        max_avg_score = 0.0

        for lang, data in lang_scores.items():
            if data['count'] > max_count:
                max_count = data['count']
                max_lang = lang
                max_avg_score = data['total_score'] / data['count']
            elif data['count'] == max_count and (data['total_score'] / data['count']) > max_avg_score:
                max_lang = lang
                max_avg_score = data['total_score'] / data['count']

        return max_lang, max_avg_score

    except Exception as e:
        print(f"Error detecting language: {e}")
        return 'en', 0.0


def is_english(text, confidence_threshold=0.8):
    """
    Check if text is English with sufficient confidence
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False

    lang, confidence = detect_language(text)
    return lang == 'en' and confidence >= confidence_threshold


def process_df_with_language_check(df, progress_description):
    """
    Process a DataFrame with language detection showing progress
    """
    df['is_english'] = False
    total = len(df)

    with tqdm(total=total, desc=progress_description, position=1, leave=False) as pbar:
        for idx in range(total):
            df.iloc[idx, df.columns.get_loc('is_english')] = is_english(df.iloc[idx]['cleaned_text'])
            pbar.update(1)

    return df[df['is_english']].drop('is_english', axis=1)


def clean_comments(df):
    """
    Clean comments dataframe
    """
    # Remove [removed] comments and AutoModerator
    df = df[df['text'] != '[deleted]']
    df = df[df['text'] != '[removed]']
    df = df[df['author'] != 'AutoModerator']

    # Rename and clean empty texts
    df = df.rename(columns={'text': 'cleaned_text'})
    df = df[df['cleaned_text'].notna() & (df['cleaned_text'] != '')]

    # Process language
    return process_df_with_language_check(df, "Checking comment languages")


def process_submissions(df):
    """
    Process submissions
    """
    df = df.copy()

    # Process text
    mask_removed = df['text'].isin(["[removed]", "[deleted]"])
    df['cleaned_text'] = None
    df.loc[mask_removed, 'cleaned_text'] = df.loc[mask_removed, 'title']
    mask_combine = ~df['text'].isin(["[removed]", "[deleted]"])
    df.loc[mask_combine, 'cleaned_text'] = df.loc[mask_combine, 'title'] + ' ' + df.loc[mask_combine, 'text']

    # Remove empty texts
    df = df[df['cleaned_text'].notna() & (df['cleaned_text'] != '')]

    # Process language
    return process_df_with_language_check(df, "Checking submission languages")


def process_subreddit_data(subreddit_path):
    """
    Process all parquet files in a subreddit directory
    """
    subreddit_path = Path(subreddit_path)

    # Process comments
    comment_dfs = []
    comment_files = list(subreddit_path.glob('comment_*.parquet'))

    for file in tqdm(comment_files, desc="Processing comment files", position=0):
        df = pd.read_parquet(file)
        initial_len = len(df)
        cleaned_df = clean_comments(df)
        comment_dfs.append(cleaned_df)
        print(f"File {file.name}: Retained {len(cleaned_df)}/{initial_len} English comments")

    # Process submissions
    submission_dfs = []
    submission_files = list(subreddit_path.glob('submission_*.parquet'))

    for file in tqdm(submission_files, desc="Processing submission files", position=0):
        df = pd.read_parquet(file)
        initial_len = len(df)
        processed_df = process_submissions(df)
        submission_dfs.append(processed_df)
        print(f"File {file.name}: Retained {len(processed_df)}/{initial_len} English submissions")

    # Combine data
    all_comments = pd.concat(comment_dfs, ignore_index=True) if comment_dfs else pd.DataFrame()
    all_submissions = pd.concat(submission_dfs, ignore_index=True) if submission_dfs else pd.DataFrame()

    return all_comments, all_submissions


def main():
    base_path = "/Users/ormeiri/Desktop/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-/data/processed"

    # Get list of subreddits
    subreddits = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Process each subreddit
    for subreddit_dir in tqdm(subreddits, desc="Processing subreddits", position=0):
        print(f"\nProcessing subreddit: {subreddit_dir}")
        subreddit_path = os.path.join(base_path, subreddit_dir)

        cleaned_comments, processed_submissions = process_subreddit_data(subreddit_path)

        # Save processed data
        output_path = os.path.join(subreddit_path, "processed")
        os.makedirs(output_path, exist_ok=True)

        cleaned_comments.to_parquet(os.path.join(output_path, "cleaned_comments.parquet"))
        processed_submissions.to_parquet(os.path.join(output_path, "processed_submissions.parquet"))

        print(f"Completed {subreddit_dir}:")
        print(f"Total retained: {len(cleaned_comments)} comments, {len(processed_submissions)} submissions")


if __name__ == "__main__":
    main()