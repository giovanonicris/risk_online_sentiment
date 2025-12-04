# shared utilities for enterprise and emerging risks scripts
# handles common functionality for both processing

import requests
import random
import re
import time
import os
import sys
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser
import datetime as dt
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse
import csv

# Load environment variables
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
MAX_ARTICLES_PER_TERM = int(os.getenv('MAX_ARTICLES_PER_TERM', '20'))

# CHUNKING - disable limit if chunking
if os.getenv('TERM_START') is not None:
    MAX_SEARCH_TERMS = None
else:
    MAX_SEARCH_TERMS = 1 if DEBUG_MODE else None

class ScraperSession:
    def __init__(self):
        self.session = self._setup_session()
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'
        ]
    
    def _setup_session(self):
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, 
                       status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session
    
    def get_random_headers(self):
        return {'User-Agent': random.choice(self.user_agents)}

# download NLTK resources if not already present
# added POS tagging (averaged_perceptron_tagger) to help in the keyword extraction which fails at times
def setup_nltk():
    for resource in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)

# extract domain name from URL
def get_source_name(url):
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.replace('www.', '')
    parts = domain.split('.')
    
    # if domain ends with country-specific TLD, take the part before it
    if len(parts) > 2 and re.match(r'^[a-z]{2}$', parts[-1]):
        return parts[-2]
    
    # for subdomains like finance.yahoo.com or markets.financialcontent.com, take the last part before .com
    if len(parts) >= 3 and parts[-1] in ('com', 'org', 'net', 'edu', 'gov'):
        # check if the second-to-last is a subdomain like 'finance' or 'markets'
        if re.match(r'^[a-z]+$', parts[-2]) and len(parts[-2]) > 3:
            return parts[-2]
        else:
            return parts[0]
        
    # for simple domains like ft.com, return full domain
    if len(parts) == 2 and parts[-1] in ('com', 'org', 'net', 'edu', 'gov'):
        return '.'.join(parts)
    
    # default: first part
    return parts[0] if parts else ''

# Dedup and load existing links from CSV
def load_existing_links(csv_path):
    if DEBUG_MODE:
        print("DEBUG: Skipping existing links check")
        return set()
    
    if not os.path.exists(csv_path):
        return set()
    
    try:
        df = pd.read_csv(csv_path, usecols=lambda x: 'LINK' in x, encoding="utf-8")
        links = set(df["LINK"].dropna().str.lower().str.strip())
        print(f"Loaded {len(links)} existing links")
        return links
    except Exception as e:
        print(f"Warning: Could not load existing links: {e}")
        return set()

# output directory setup
def setup_output_dir(output_csv):
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    return output_dir / output_csv

# Load and validate search terms from CSV
def load_search_terms(encoded_csv_path, risk_id_col):
    try:
        usecols = [risk_id_col, 'SEARCH_TERM_ID', 'ENCODED_TERMS']
        df = pd.read_csv(f'data/{encoded_csv_path}', encoding='utf-8', usecols=usecols)
        df[risk_id_col] = pd.to_numeric(df[risk_id_col], downcast='integer', errors='coerce')
        print(f"Loaded {len(df)} search terms from {encoded_csv_path}")
        return df
    except FileNotFoundError:
        print(f"ERROR!!! data/{encoded_csv_path} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading data/{encoded_csv_path}: {e}")
        sys.exit(1)

# save results to CSV with deduplication and archiving
def save_results(df, output_path, risk_type):
    # save results to csv with deduplication per risk_id
    print(f"Saving {len(df)} {risk_type} articles to {output_path}")
    
    # load existing data
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, parse_dates=['PUBLISHED_DATE'], encoding='utf-8')
        print(f"Loaded existing CSV with {len(existing_df)} records")
    else:
        existing_df = pd.DataFrame()
        print("No existing CSV found - starting fresh")
    
    # combine and dedup by risk_id, title, and link
    combined_df = pd.concat([existing_df, df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['RISK_ID', 'TITLE', 'LINK'], keep='first')
    
    # 4-month rolling window
    cutoff_date = dt.datetime.now() - dt.timedelta(days=4 * 30)
    combined_df['PUBLISHED_DATE'] = pd.to_datetime(combined_df['PUBLISHED_DATE'], errors='coerce')
    
    current_df = combined_df[combined_df['PUBLISHED_DATE'] >= cutoff_date].copy()
    old_df = combined_df[combined_df['PUBLISHED_DATE'] < cutoff_date].copy()
    
    # save current data
    current_df.sort_values(by='PUBLISHED_DATE', ascending=False).to_csv(
        output_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL
    )
    
    print(f"Updated main CSV with {len(current_df)} records")
    
    # Archive old data (skip in debug)
    if not DEBUG_MODE and not old_df.empty:
        archive_path = Path(output_path).parent / f'{risk_type}_sentiment_archive.csv'
        old_df.sort_values(by='PUBLISHED_DATE').to_csv(
            archive_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL
        )
        print(f"Archived {len(old_df)} records")
    
    return len(current_df)

# print debug info
def print_debug_info(script_name, risk_type, start_time):
    print("*" * 50)
    print(f"{script_name} - {risk_type.upper()} News Sentiment Scraper")
    print(f"DEBUG_MODE: {DEBUG_MODE}")
    if DEBUG_MODE:
        print(f"   - Max terms: {MAX_SEARCH_TERMS}")
        print(f"   - Max articles per term: {MAX_ARTICLES_PER_TERM}")
    print(f"Script started: {start_time}")
    print(f"Working directory: {os.getcwd()}")
    print("*" * 50)

# load whitelist and source type lists
def load_source_lists():
    try:
        # load source and type data
        source_df = pd.read_csv('data/source_and_type.csv', encoding='utf-8')
        whitelist = set(source_df[source_df['CREDIBILITY_TYPE'] == 'Mainstream']['SOURCE_NAME'].str.lower().str.strip())
        paywalled = set(source_df[source_df['IS_PAYWALLED'] == 1]['SOURCE_NAME'].str.lower().str.strip())
        credibility_map = dict(zip(source_df['SOURCE_NAME'].str.lower().str.strip(), source_df['CREDIBILITY_TYPE']))
        # load curated whitelist from filter_in_sources.csv
        exclusive_whitelist = set()
        try:
            filter_df = pd.read_csv('data/filter_in_sources.csv', encoding='utf-8')
            exclusive_whitelist = set(filter_df['SOURCE_NAME'].str.lower().str.strip())
            print(f"loaded {len(exclusive_whitelist)} exclusive whitelist sources")
        except FileNotFoundError:
            print("warning: data/filter_in_sources.csv not found, using empty exclusive set")
        except Exception as e:
            print(f"warning loading exclusive whitelist: {e}")
        
        print(f"Loaded {len(whitelist)} whitelist sources")
        print(f"Loaded {len(paywalled)} paywalled sources")
        print(f"Loaded {len(credibility_map)} credibility mappings")
        print(f"Loaded {len(exclusive_whitelist)} exclusive whitelist sources")

        return whitelist, paywalled, credibility_map, exclusive_whitelist
    except Exception as e:
        print(f"Warning: Could not load source lists: {e}")
        return set(), set(), {}

# calculate quality score for an article
def calculate_quality_score(title, summary, source_url, search_terms, whitelist):
    scores = {
        'relevance': 0, 'recency': 0, 'length_150': 0, 'length_500': 0,
        'whitelist_bonus': 0, 'clickbait_penalty': 0, 'total_score': 0
    }
    
    # basic relevance check
    title_lower = str(title).lower()
    summary_lower = str(summary).lower() if summary else ''
    text = f"{title_lower} {summary_lower}"
    
    # check relevance to search terms
    relevant_terms = sum(1 for term in search_terms if re.search(re.escape(str(term).lower()), text))
    scores['relevance'] = min(relevant_terms, 2)  # cap at 2
    
    # recency (you'll need to pass publish date)
    scores['recency'] = 1
    
    # source quality - IF IN WHITELIST
    if source_url:
        source_name = get_source_name(source_url)
        if any(white in source_name for white in whitelist):
            scores['whitelist_bonus'] = 2
    
    # clickbait detection (hard-coded basic patterns)
    clickbait_patterns = ['clickbait', 'shocking', 'unbelievable', 'you won\'t believe']
    scores['clickbait_penalty'] = -2 if any(re.search(pattern, title_lower) for pattern in clickbait_patterns) else 0
    
    # content length
    # NOTE: summary articles of video news will have very short text and is considered low quality because it's not the full article
    article_text = f"{title} {summary}" if summary else title
    word_count = len(str(article_text).split())
    scores['length_150'] = 1 if word_count > 150 else 0
    scores['length_500'] = 1 if word_count > 500 else 0

    # calculate total
    total_score = sum(scores.values())
    scores['total_score'] = max(total_score, 0)
    
    return scores