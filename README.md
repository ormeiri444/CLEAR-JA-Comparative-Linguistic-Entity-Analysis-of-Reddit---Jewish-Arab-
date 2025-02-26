# CLEAR-JA: Comparative Linguistic Entity Analysis of Reddit - Jewish-Arab

## Project Overview
CLEAR-JA is a research project that performs comparative linguistic analysis of Reddit content related to Jewish and Arab topics. The project uses Natural Language Processing (NLP) techniques to identify entities, analyze sentiment, and extract insights from Reddit discussions.

## Features
- Named Entity Recognition (NER) to identify people, organizations, and locations
- Sentiment analysis of Reddit comments and posts
- Comparative analysis between Jewish and Arab discourse on Reddit
- Data visualization of linguistic patterns and trends

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
1. Clone the repository:
```bash
git clone https://github.com/ormeiri444/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-.git
cd CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables for API access:
```bash
# Create a .env file and add your API keys
touch .env
# Add the following to your .env file
HF_API_TOKEN=your_hugging_face_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```
Note: Never commit your .env file to version control.

## Usage

### Data Collection
```bash
python src/data_collection/reddit_scraper.py --subreddit "subreddit_name" --limit 100
```

### NER and Sentiment Analysis
```bash
python src/NER_and_Sentiment/NER_and_sentiment_label.py --input "input_file.csv" --output "output_file.csv"
```

### Visualization
```bash
python src/visualization/generate_plots.py --input "analyzed_data.csv" --output "output_directory"
```

## Project Structure
```
CLEAR-JA/
├── data/                  # Data storage
│   ├── raw/               # Raw scraped data
│   ├── processed/         # Processed data after analysis
│   └── results/           # Analysis results
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── src/                   # Source code
│   ├── data_collection/   # Code for collecting Reddit data
│   ├── NER_and_Sentiment/ # Named entity recognition and sentiment analysis
│   ├── analysis/          # Code for comparative analysis
│   └── visualization/     # Visualization tools
├── README.md              # This file
└── requirements.txt       # Dependencies
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
[MIT License](LICENSE)

## Contact
- Project Maintainer: Or Meiri - ormeiri123@gmail.com
- Project Link: [https://github.com/ormeiri444/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-](https://github.com/ormeiri444/CLEAR-JA-Comparative-Linguistic-Entity-Analysis-of-Reddit---Jewish-Arab-)

## Acknowledgements
- [Hugging Face](https://huggingface.co/) for NLP models
- [PRAW](https://praw.readthedocs.io/en/stable/) for Reddit API access
- [spaCy](https://spacy.io/) for NLP processing
