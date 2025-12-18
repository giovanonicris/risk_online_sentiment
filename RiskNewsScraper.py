# RISK NEWS SCRAPER
# unified script for enterprise and emerging risks; uses shared utilities for common functionality; this includes the debug mode

import datetime as dt
import random
import time
import re
import csv
import requests
from pathlib import Path
from newspaper import Article, Config
from googlenewsdecoder import new_decoderv1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser
import sys
from keybert import KeyBERT
import xml.etree.ElementTree as ET
import argparse
import os
import base64
from gnews import GNews

# GLOBAL CONSTANTS
SEARCH_DAYS = 7  # look back this many days for news articles; edit to change

# decoding logic (retained from original script but made a fx)
def process_encoded_search_terms(term):
    try:
        encoded_number = int(term)
        byte_length = (encoded_number.bit_length() + 7) // 8
        byte_rep = encoded_number.to_bytes(byte_length, byteorder='little')
        decoded_text = byte_rep.decode('utf-8')
        return decoded_text
    except (ValueError, UnicodeDecodeError, OverflowError):
        return None

# IMPORTANT!! Import shared utilities from utils.py
from utils import (
    ScraperSession, setup_nltk, load_existing_links, setup_output_dir,
    save_results, print_debug_info, DEBUG_MODE,
    MAX_ARTICLES_PER_TERM, MAX_SEARCH_TERMS, load_source_lists, 
    calculate_quality_score, get_source_name
)

# setup argparse for risk type (no chunking anymore)
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--risk-type', type=str, choices=['enterprise', 'emerging'], default='enterprise')
args = arg_parser.parse_args()

# this is the main fx that orchestrates the entire process.
def main():
    # config based on risk type
    risk_type = args.risk_type
    if risk_type == "enterprise":
        risk_id_col = "ENTERPRISE_RISK_ID"
        ENCODED_CSV = "EnterpriseRisksListEncoded.csv"
        OUTPUT_CSV = "enterprise_risks_online_sentiment.csv"
    else:  # emerging
        risk_id_col = "EMERGING_RISK_ID"
        ENCODED_CSV = "EmergingRisksListEncoded.csv"
        OUTPUT_CSV = "emerging_risks_online_sentiment.csv"
    
    RISK_TYPE = risk_type  # for logging/saving
    
    # process time start for reference
    print("*" * 50)
    start_time = dt.datetime.now()
    print_debug_info("RiskNewsScraper", RISK_TYPE, start_time)

    # setup NLTK and session etc.
    setup_nltk()
    session = ScraperSession()
    analyzer = SentimentIntensityAnalyzer()
    
    # load data
    output_path = setup_output_dir(OUTPUT_CSV)
    existing_links = load_existing_links(output_path)
    search_terms_df = load_search_terms(ENCODED_CSV, risk_id_col)
    
    # only limit search terms in debug mode
    if DEBUG_MODE and MAX_SEARCH_TERMS:
        search_terms_df = search_terms_df.head(MAX_SEARCH_TERMS)
        print(f"DEBUG: Limited to first {MAX_SEARCH_TERMS} search terms")
    
    # load whitelist, paywalled, and credibility sources
    whitelist, paywalled, credibility_map, exclusive_whitelist = load_source_lists()
    
    # process articles
    articles_df = process_risk_articles(search_terms_df, session, existing_links, analyzer, whitelist, paywalled, credibility_map, exclusive_whitelist, risk_id_col)
    print(f"Processed DF size: {len(articles_df)}") # debug print
    
    # save results
    if not articles_df.empty:
        record_count = save_results(articles_df, output_path, RISK_TYPE)
        print(f"About to save to: {str(output_path)}") # debug print
        print(f"Completed: {record_count} total records") # validation print
    else:
        print("WARNING!!! No articles processed!!")
    
    # end time for reference
    print(f"Completed at: {dt.datetime.now()}")
    print("*" * 50)

def load_search_terms(encoded_csv_path, risk_id_col):
    # load and decode search terms from CSV - ORIGINAL LOGIC
    try:
        usecols = [risk_id_col, 'SEARCH_TERM_ID', 'ENCODED_TERMS']
        df = pd.read_csv(f'data/{encoded_csv_path}', encoding='utf-8', usecols=usecols)
        df[risk_id_col] = pd.to_numeric(df[risk_id_col], downcast='integer', errors='coerce')
        
        # ORIGINAL DECODING LOGIC
        df['SEARCH_TERMS'] = df['ENCODED_TERMS'].apply(process_encoded_search_terms)

        print(f"Loaded {len(df)} search terms from {encoded_csv_path}")
        valid_terms = df['SEARCH_TERMS'].dropna()
        print(f"Valid search terms ({len(valid_terms)}): {valid_terms.head().tolist()}")
        
        # filter out rows with invalid search terms
        valid_df = df.dropna(subset=['SEARCH_TERMS'])
        if valid_df.empty:
            print("ERROR!!! No valid search terms after decoding!!")
            sys.exit(1)
        return valid_df
    except FileNotFoundError:
        print(f"ERROR!!! data/{encoded_csv_path} not found!!")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading data/{encoded_csv_path}: {e}")
        sys.exit(1)

def process_risk_articles(search_terms_df, session, existing_links, analyzer, whitelist, paywalled, credibility_map, exclusive_whitelist, risk_id_col):
    # this is the MAIN processing loop for risk articles
    print(f"Processing {len(search_terms_df)} search terms...")
    
    all_articles = []
    
    # setup newspaper config
    config = Config()
    user_agent = random.choice(session.user_agents)
    config.browser_user_agent = user_agent
    config.enable_image_fetching = False  # faster without images!
    config.request_timeout = 10 if DEBUG_MODE else 20
    
    # set dates for search (using SEARCH_DAYS global constant)
    now = dt.date.today()
    yesterday = now - dt.timedelta(days=SEARCH_DAYS)
    
# process each search term
    # helper for single term processing - for parallel
    def process_single_term(row, risk_id_col):
        search_term = row['SEARCH_TERMS']
        risk_id = row[risk_id_col]
        search_term_id = row['SEARCH_TERM_ID']
        
        if pd.isna(search_term):
            print(f"  ---skipping invalid search term for risk ID {risk_id}")
            return []
            
        print(f"processing search term (ID: {risk_id}, SEARCH_TERM_ID: {search_term_id}) - '{search_term[:50]}...'")  # dropped idx since parallel
        
        # Get Google News articles
        articles = get_google_news_articles(search_term, session, existing_links, MAX_ARTICLES_PER_TERM, now, yesterday, whitelist, paywalled, credibility_map, exclusive_whitelist)
        
        # PRETTY SOURCE NAME
        # parse raw rss for source names (google news decoder doesn't expose it)
        try:
            # refetch rss for this term's first page to get xml
            rss_url = f'https://news.google.com/rss/search?q={search_term}%20when%3A{SEARCH_DAYS}d&start=0'
            req_rss = session.session.get(rss_url, headers=session.get_random_headers())
            rss_xml = req_rss.content.decode('utf-8')
            root = ET.fromstring(rss_xml)
            items = root.findall('.//item')
            source_dict = {}
            for i, item in enumerate(items):
                source_elem = item.find('source')
                if source_elem is not None:
                    source_dict[i] = source_elem.text.strip()  # e.g., "Financial Times"
            # map to articles by index (assuming order matches)
            for j, art in enumerate(articles):
                if j in source_dict:
                    art['pretty_source'] = source_dict[j]
        except Exception as e:
            print(f"rss source parse failed: {e}")
            # fallback: no change

        if not articles:
            print(f"  - No new articles found for this term")
            return []

        # process batch
        processed = process_articles_batch(articles, config, analyzer, search_term, whitelist, risk_id, search_term_id, existing_links)
        print(f"  ---processed {len(processed)} articles")
        return processed
    
    # parallel process terms
    if DEBUG_MODE and len(search_terms_df) > 1:
        # debug: process sequentially
        for _, row in search_terms_df.iterrows():
            term_articles = process_single_term(row, risk_id_col)
            all_articles.extend(term_articles)
            if len(all_articles) >= 5:  # debug limit
                break
    else:
        # full parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_single_term, row, risk_id_col) for _, row in search_terms_df.iterrows()]
            for future in futures:
                all_articles.extend(future.result())
    
    if all_articles:
        articles_df = pd.DataFrame(all_articles)
        if DEBUG_MODE:
            print(f"DEBUG: Total articles before dedup: {len(articles_df)}")
        return articles_df
    else:
        print("No articles to process")
        return pd.DataFrame()
    
import requests

def get_google_news_articles(search_term, session, existing_links, max_articles, now, yesterday, whitelist, paywalled, credibility_map, exclusive_whitelist):
    api_key = os.getenv('GNEWS_API_KEY')
    if not api_key:
        print("ERROR: GNEWS_API_KEY not set or empty!")
        return []

    # chunk the whitelist to avoid URL too long
    chunk_size = 10
    whitelist_chunks = [list(exclusive_whitelist)[i:i + chunk_size] for i in range(0, len(exclusive_whitelist), chunk_size)]
    print(f"using {len(whitelist_chunks)} whitelist chunks of size {chunk_size}")

    all_articles = []
    seen_urls = set()  # dedup across chunks

    for chunk_idx, chunk in enumerate(whitelist_chunks):
        site_query = " OR ".join(f"site:{domain}" for domain in chunk)
        query = f"{search_term} ({site_query})" if site_query else search_term

        url = "https://gnews.io/api/v4/search"
        params = {
            'q': query,
            'lang': 'en',
            'country': 'us',
            'max': max_articles,
            'apikey': api_key
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            time.sleep(1)  # avoid burst rate limit in API
            if resp.status_code != 200:
                print(f"GNews chunk {chunk_idx+1} failed: {resp.status_code} {resp.text}")
                continue
            data = resp.json()
        except Exception as e:
            print(f"GNews chunk {chunk_idx+1} request failed: {e}")
            continue

        for item in data.get('articles', []):
            url = item['url']
            if url.lower().strip() in existing_links or url in seen_urls:
                continue

            seen_urls.add(url)

            title = item['title']
            source_text = item['source']['name']

            parsed_url = urlparse(url)
            full_domain = parsed_url.netloc.lower().replace('www.', '')

            google_index = len(all_articles) + 1

            is_paywalled = full_domain.lower() in paywalled
            credibility_type = credibility_map.get(full_domain.lower(), 'Relevant Article')

            all_articles.append({
                'url': url,
                'title': title,
                'html': None,
                'google_index': google_index,
                'paywalled': is_paywalled,
                'credibility_type': credibility_type
            })
            print(f"    - Added article: '{title[:50]}...' from {source_text} (domain: {get_source_name(url)}, full_domain: {full_domain})")

            if len(all_articles) >= max_articles:
                print(f"  ---found {len(all_articles)} new articles via GNews (chunked)")
                return all_articles

    print(f"  ---found {len(all_articles)} new articles via GNews (chunked)")
    return all_articles

def process_articles_batch(articles, config, analyzer, search_term, whitelist, risk_id, search_term_id, existing_links): #STID to delete later!
    # Process in parallel for optimization...
    processed = []
    seen_urls = set()  # DEDUP LAYER - track urls for this search term
    seen_titles = set()  # DEDUP LAYER - track titles for this search term
    
    def process_single_article(article_data):
        # handle single article processing
        try:
            url = article_data['url']
            title = article_data['title']
            google_index = article_data.get('google_index', 0)  # get index from article to see the sort order
            is_paywalled = article_data.get('paywalled', False)
            credibility_type = article_data.get('credibility_type', 'Relevant Article')
            
            # deduplicate by url and title for this search term
            url_key = url.lower().strip()
            title_key = title.lower().strip()[:100]  # limit title length for comparison
            if url_key in seen_urls or title_key in seen_titles:
                if DEBUG_MODE:
                    print(f"  ---Skipping duplicate: '{title[:50]}...' ({url[:50]}...)")
                return None
            seen_urls.add(url_key)
            seen_titles.add(title_key)
            
            # removed existing_links check to handle in save_results - DEDUP LOGIC HANDLED in utils.py
            # if url.lower().strip() in existing_links:
            #     return None
            
            # PRE-FILTER: Skip known problematic URL patterns from manual review
            # Add as needed based on result review
            problematic_patterns = [
                '/video/', '/videos/', '/watch/',
                'wsj.com/subscriptions', 'bloomberg.com/newsletters',
                'reuters.com/video', 'reuters.com/graphics'
            ]
            
            if any(pattern in url.lower() for pattern in problematic_patterns):
                if DEBUG_MODE:
                    print(f"  - Skipping problematic URL: {title[:50]}... ({url[:50]}...)")
                return None
            
            # download and parse article
            article = Article(url, config=config)
            article.download()
            
            # check if download succeeded - FIXED: Use try/except instead of download_exception
            if not article.html or article.html.strip() == '':
                if DEBUG_MODE:
                    print(f"  ---Download failed for '{title[:50]}...' (empty HTML)")
                return None
                
            #parse article, extract keywords    
            article.parse()
            keywords = article.keywords if article.keywords else []
            # KEYWORD EXTRACT FALLBACK - use KeyBERT if no keywords found using newspaper lib
            if not keywords and article.text:
                kw_model = KeyBERT()
                keywords = kw_model.extract_keywords(article.text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
                keywords = [kw[0] for kw in keywords]  # extract keyword strings
            if DEBUG_MODE:
                print(f"    - Extracted keywords for '{title[:50]}...': {keywords}")
                print(f"    - Article text length: {len(article.text) if article.text else 0} chars")
            
            # extract content
            summary = article.summary if article.summary else article.text[:500]
            
            # skip empty content
            if not summary or len(summary.strip()) < 50:
                if DEBUG_MODE:
                    print(f"  ---Empty content for '{title[:50]}...'")
                return None
            
            # sentiment analysis
            sentiment = analyzer.polarity_scores(title + " " + summary)
            sentiment_category = 'Negative' if sentiment['compound'] <= -0.05 else 'Positive' if sentiment['compound'] >= 0.05 else 'Neutral'
            
            # quality scoring
            quality_scores = calculate_quality_score(
                title, summary, url, [search_term], whitelist
            )
            
            # include all articles, keeping quality score for review
            print(f"DEBUG: Assigning SEARCH_TERM_ID={search_term_id} to article '{title[:50]}...' (RISK_ID={risk_id})") #STID to delete later!

            # PRETTY SOURCE NAME
            # final formatting before write
            # source_name = get_source_name(url).capitalize()
            source_name = article_data.get('pretty_source', get_source_name(url)).capitalize()
            # article_data is the local var - use it for pretty_source fallback
            
            publish_date = article.publish_date or dt.datetime.now()
            formatted_publish_date = pd.to_datetime(publish_date).strftime('%Y-%m-%d %H:%M:%S')

            return {
                'RISK_ID': risk_id,  # proper risk id mapping
                'SEARCH_TERM_ID': search_term_id,  #STID to delete later!
                'GOOGLE_INDEX': google_index,  # google news position for this article
                'TITLE': title,
                'LINK': url,
                'PUBLISHED_DATE': formatted_publish_date,
                'SUMMARY': summary[:500],  # truncate for CSV size
                'KEYWORDS': ', '.join(keywords) if keywords else '',
                'SENTIMENT_COMPOUND': sentiment['compound'],
                'SENTIMENT': sentiment_category,
                'SOURCE': source_name,
                'SOURCE_URL': url,
                'PAYWALLED': is_paywalled,
                'CREDIBILITY_TYPE': credibility_type,
                'QUALITY_SCORE': quality_scores['total_score'],
                # add individual score components
                **{f'SCORE_{k.upper()}': v for k, v in quality_scores.items() if k != 'total_score'},
            }
                
        except Exception as e:
            if DEBUG_MODE:
                print(f"  ---error processing article '{title[:50] if 'title' in locals() else 'Unknown'}...': {e}")
            return None
    
    # process with threading (limit to 3 concurrent to avoid overload)
    if articles:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = executor.map(process_single_article, articles)
            processed = [r for r in results if r is not None]
    
    return processed

if __name__ == '__main__':
    main()
