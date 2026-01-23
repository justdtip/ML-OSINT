#!/usr/bin/env python3
"""
Ukraine War Timeline Scraper

Scrapes structured timeline data from multiple sources:
1. Wikipedia Timeline pages (via MediaWiki API)
2. Wikipedia List of Military Engagements (tables)
3. Individual Wikipedia battle articles for dates

Usage:
    python scrape_war_timelines.py [--source SOURCE] [--output-dir DIR]
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('timeline_scraping.log')
    ]
)
logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent

# Wikipedia API endpoint
WIKI_API = "https://en.wikipedia.org/w/api.php"

# Timeline page URLs
WIKIPEDIA_TIMELINE_PAGES = [
    "Timeline_of_the_Russian_invasion_of_Ukraine",
    "Timeline_of_the_Russian_invasion_of_Ukraine_(24_February_–_7_April_2022)",
    "Timeline_of_the_Russian_invasion_of_Ukraine_(8_April_–_28_August_2022)",
    "Timeline_of_the_Russian_invasion_of_Ukraine_(29_August_–_11_November_2022)",
    "Timeline_of_the_Russian_invasion_of_Ukraine_(12_November_2022_–_7_June_2023)",
    "Timeline_of_the_Russian_invasion_of_Ukraine_(1_September_–_30_November_2023)",
    "Timeline_of_the_Russian_invasion_of_Ukraine_(1_December_2023_–_31_March_2024)",
    "Timeline_of_the_Russian_invasion_of_Ukraine_(1_April_–_31_July_2024)",
    "Timeline_of_the_Russian_invasion_of_Ukraine_(1_August_–_31_December_2024)",
    "Timeline_of_the_Russian_invasion_of_Ukraine_(1_January_2025_–_31_May_2025)",
    "Timeline_of_the_Russian_invasion_of_Ukraine_(1_June_2025_–_present)",
]

WIKIPEDIA_ENGAGEMENTS_PAGE = "List_of_military_engagements_during_the_Russo-Ukrainian_war_(2022–present)"

# Month name mapping
MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}


class WikipediaTimelineScraper:
    """Scrapes timeline events from Wikipedia pages."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ML-OSINT-Research/1.0 (Academic research project)'
        })

    def fetch_page_html(self, page_title: str) -> Optional[str]:
        """Fetch parsed HTML content for a Wikipedia page."""
        params = {
            'action': 'parse',
            'page': page_title,
            'format': 'json',
            'prop': 'text',
            'disabletoc': 'true'
        }

        try:
            response = self.session.get(WIKI_API, params=params)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                logger.error(f"API error for {page_title}: {data['error']}")
                return None

            return data['parse']['text']['*']
        except Exception as e:
            logger.error(f"Failed to fetch {page_title}: {e}")
            return None

    def fetch_page_wikitext(self, page_title: str) -> Optional[str]:
        """Fetch raw wikitext for a page."""
        params = {
            'action': 'query',
            'titles': page_title,
            'prop': 'revisions',
            'rvprop': 'content',
            'rvslots': 'main',
            'format': 'json'
        }

        try:
            response = self.session.get(WIKI_API, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            if page_id == '-1':
                return None

            return pages[page_id]['revisions'][0]['slots']['main']['*']
        except Exception as e:
            logger.error(f"Failed to fetch wikitext for {page_title}: {e}")
            return None

    def parse_timeline_wikitext(self, wikitext: str, page_title: str) -> list[dict]:
        """Parse timeline events from wikitext format."""
        events = []

        # Match patterns like "* '''24 February''' – text"
        # or "** [[Battle name]] – text"
        lines = wikitext.split('\n')

        current_year = None
        current_month = None

        # Extract year from page title
        year_match = re.search(r'(\d{4})', page_title)
        if year_match:
            current_year = int(year_match.group(1))

        for line in lines:
            line = line.strip()

            # Skip empty lines and headers
            if not line or line.startswith('='):
                continue

            # Check for month headers (== February == or similar)
            month_header = re.match(r"^==+\s*(\w+)\s*==+$", line)
            if month_header:
                month_name = month_header.group(1).lower()
                if month_name in MONTHS:
                    current_month = MONTHS[month_name]
                continue

            # Check for year headers
            year_header = re.match(r"^==+\s*(\d{4})\s*==+$", line)
            if year_header:
                current_year = int(year_header.group(1))
                continue

            # Match list items with dates
            # Pattern: * '''24 February''' – or * '''24''' – (within month section)
            date_patterns = [
                r"^\*+\s*'''?(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)'''?\s*[–—-]\s*(.+)",
                r"^\*+\s*'''?(\d{1,2})'''?\s*[–—-]\s*(.+)",  # Day only within month section
            ]

            for i, pattern in enumerate(date_patterns):
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if i == 0:  # Full date pattern
                        day = int(match.group(1))
                        month_name = match.group(2).lower()
                        month = MONTHS.get(month_name)
                        text = match.group(3)
                    else:  # Day only pattern
                        day = int(match.group(1))
                        month = current_month
                        text = match.group(2)

                    if month and current_year and day:
                        try:
                            date_obj = datetime(current_year, month, day)
                            iso_date = date_obj.strftime("%Y-%m-%d")
                        except ValueError:
                            iso_date = f"{current_year}-{month:02d}-{day:02d}"

                        # Clean text
                        text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', text)  # [[link|text]] -> text
                        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[link]] -> link
                        text = re.sub(r"'''?", '', text)  # Remove bold
                        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text)  # Remove refs
                        text = re.sub(r'<ref[^/]*/>', '', text)  # Remove self-closing refs
                        text = text.strip()

                        if len(text) > 20:  # Only meaningful events
                            events.append({
                                'date': iso_date,
                                'text': text[:500],  # Truncate very long
                                'source': f"Wikipedia:{page_title}",
                                'type': 'timeline_event'
                            })
                    break

        return events

    def parse_timeline_html(self, html: str, page_title: str) -> list[dict]:
        """Parse timeline events from HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        events = []

        # Extract year from page title
        current_year = None
        year_match = re.search(r'(\d{4})', page_title)
        if year_match:
            current_year = int(year_match.group(1))

        current_month = None

        # Find all headers and list items
        for element in soup.find_all(['h2', 'h3', 'h4', 'li']):
            text = element.get_text(strip=True)

            # Check for month/year headers
            if element.name in ['h2', 'h3', 'h4']:
                # Month header
                for month_name, month_num in MONTHS.items():
                    if month_name in text.lower():
                        current_month = month_num
                        break

                # Year in header
                year_in_text = re.search(r'\b(20\d{2})\b', text)
                if year_in_text:
                    current_year = int(year_in_text.group(1))
                continue

            # Parse list items for events
            if element.name == 'li':
                # Look for date at start of text
                date_match = re.match(
                    r'^(\d{1,2})\s*(January|February|March|April|May|June|July|August|September|October|November|December)?\s*[–—:-]?\s*(.+)',
                    text, re.IGNORECASE
                )

                if date_match and len(text) > 30:
                    day = int(date_match.group(1))
                    month_name = date_match.group(2)
                    event_text = date_match.group(3)

                    if month_name:
                        month = MONTHS.get(month_name.lower())
                    else:
                        month = current_month

                    if month and current_year:
                        try:
                            date_obj = datetime(current_year, month, day)
                            iso_date = date_obj.strftime("%Y-%m-%d")

                            # Clean text
                            event_text = re.sub(r'\[\d+\]', '', event_text)  # Remove citations
                            event_text = event_text.strip()

                            if len(event_text) > 20:
                                events.append({
                                    'date': iso_date,
                                    'text': event_text[:500],
                                    'source': f"Wikipedia:{page_title}",
                                    'type': 'timeline_event'
                                })
                        except ValueError:
                            pass

        return events

    def scrape_all_timelines(self) -> list[dict]:
        """Scrape all Wikipedia timeline pages."""
        all_events = []

        for page_title in WIKIPEDIA_TIMELINE_PAGES:
            logger.info(f"Scraping: {page_title}")

            # Try wikitext first (more reliable parsing)
            wikitext = self.fetch_page_wikitext(page_title)
            if wikitext:
                events = self.parse_timeline_wikitext(wikitext, page_title)
                if events:
                    all_events.extend(events)
                    logger.info(f"  Extracted {len(events)} events (wikitext)")
                    time.sleep(1)
                    continue

            # Fallback to HTML parsing
            html = self.fetch_page_html(page_title)
            if html:
                events = self.parse_timeline_html(html, page_title)
                all_events.extend(events)
                logger.info(f"  Extracted {len(events)} events (HTML)")

            time.sleep(1)

        return all_events

    def scrape_engagements_with_dates(self) -> list[dict]:
        """Scrape military engagements and fetch dates from individual articles."""
        logger.info(f"Scraping engagements: {WIKIPEDIA_ENGAGEMENTS_PAGE}")

        html = self.fetch_page_html(WIKIPEDIA_ENGAGEMENTS_PAGE)
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        engagements = []

        # Find all links that look like battle/siege/offensive articles
        battle_pattern = re.compile(
            r'(Battle|Siege|Offensive|Attack|Capture|Liberation|Assault|Defence|Defense|Counteroffensive|Operation)',
            re.IGNORECASE
        )

        seen_titles = set()

        # Look for links in the page
        for link in soup.find_all('a', href=True):
            title = link.get('title', '')
            text = link.get_text(strip=True)

            if not title or title in seen_titles:
                continue

            # Skip certain patterns
            if 'Wikipedia:' in title or 'Template:' in title or 'File:' in title:
                continue

            # Match battle-related articles
            if battle_pattern.search(title) or battle_pattern.search(text):
                if 'Ukraine' in title or '2022' in title or '2023' in title or '2024' in title or '2025' in title:
                    seen_titles.add(title)
                    engagements.append({
                        'name': text if text else title,
                        'wiki_title': title,
                        'source': f"Wikipedia:{WIKIPEDIA_ENGAGEMENTS_PAGE}",
                        'type': 'military_engagement'
                    })

        logger.info(f"  Found {len(engagements)} battle articles to process")

        # Fetch dates for each engagement (with rate limiting)
        for i, eng in enumerate(engagements):
            if i % 10 == 0:
                logger.info(f"  Processing {i+1}/{len(engagements)}")

            date_info = self.fetch_battle_dates(eng['wiki_title'])
            if date_info:
                eng.update(date_info)

            # Rate limiting
            time.sleep(0.5)

        return engagements

    def fetch_battle_dates(self, wiki_title: str) -> Optional[dict]:
        """Fetch date information from a battle article's infobox."""
        wikitext = self.fetch_page_wikitext(wiki_title)
        if not wikitext:
            return None

        # Parse infobox for date
        # Pattern: | date = 24 February – 31 March 2022
        date_match = re.search(
            r'\|\s*date\s*=\s*([^\n|]+)',
            wikitext, re.IGNORECASE
        )

        result = {}

        if date_match:
            date_str = date_match.group(1).strip()
            # Clean wiki markup
            date_str = re.sub(r'\{\{[^}]+\}\}', '', date_str)
            date_str = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', date_str)
            date_str = re.sub(r'\[\[([^\]]+)\]\]', r'\1', date_str)
            date_str = re.sub(r'<[^>]+>', '', date_str)
            date_str = date_str.strip()

            result['date_raw'] = date_str

            # Try to parse start date
            start_match = re.match(
                r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
                date_str, re.IGNORECASE
            )
            if start_match:
                day = int(start_match.group(1))
                month = MONTHS.get(start_match.group(2).lower())
                year = int(start_match.group(3))
                try:
                    result['date_start'] = datetime(year, month, day).strftime("%Y-%m-%d")
                except ValueError:
                    pass

        # Parse location
        location_match = re.search(
            r'\|\s*(?:place|location)\s*=\s*([^\n|]+)',
            wikitext, re.IGNORECASE
        )
        if location_match:
            loc = location_match.group(1).strip()
            loc = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', loc)
            loc = re.sub(r'\[\[([^\]]+)\]\]', r'\1', loc)
            loc = re.sub(r'\{\{[^}]+\}\}', '', loc)
            loc = re.sub(r'<[^>]+>', '', loc)
            result['location'] = loc.strip()[:200]

        # Parse result
        result_match = re.search(
            r'\|\s*result\s*=\s*([^\n|]+)',
            wikitext, re.IGNORECASE
        )
        if result_match:
            res = result_match.group(1).strip()
            res = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', res)
            res = re.sub(r'\[\[([^\]]+)\]\]', r'\1', res)
            res = re.sub(r'\{\{[^}]+\}\}', '', res)
            result['result'] = res.strip()[:200]

        return result if result else None


class TimelineConsolidator:
    """Consolidates all timeline data into unified format."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def consolidate(
        self,
        timeline_events: list[dict],
        engagements: list[dict]
    ) -> dict:
        """Consolidate all data sources into unified format."""

        # Create date-indexed structure
        by_date = {}

        # Process timeline events
        for event in timeline_events:
            date = event.get('date', 'unknown')
            if date not in by_date:
                by_date[date] = {'events': [], 'engagements': []}
            by_date[date]['events'].append(event)

        # Process engagements
        for eng in engagements:
            date = eng.get('date_start', 'unknown')
            if date not in by_date:
                by_date[date] = {'events': [], 'engagements': []}
            by_date[date]['engagements'].append(eng)

        # Build consolidated output
        consolidated = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'sources': [
                    'Wikipedia Timeline pages',
                    'Wikipedia List of Military Engagements'
                ],
                'total_dates': len(by_date),
                'total_events': len(timeline_events),
                'total_engagements': len(engagements)
            },
            'by_date': by_date,
            'all_events': timeline_events,
            'all_engagements': engagements
        }

        return consolidated

    def save(self, data: dict, filename: str = "consolidated_timeline.json"):
        """Save consolidated data to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved consolidated data to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Ukraine war timeline data from multiple sources"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for scraped data"
    )
    parser.add_argument(
        "--source",
        choices=['all', 'wikipedia', 'engagements'],
        default='all',
        help="Which sources to scrape"
    )
    parser.add_argument(
        "--skip-consolidate",
        action='store_true',
        help="Skip consolidation step"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timeline_events = []
    engagements = []

    wiki_scraper = WikipediaTimelineScraper(args.output_dir)

    # Scrape Wikipedia timelines
    if args.source in ['all', 'wikipedia']:
        timeline_events = wiki_scraper.scrape_all_timelines()

        # Save raw timeline events
        with open(args.output_dir / "wikipedia_timeline_events.json", 'w', encoding='utf-8') as f:
            json.dump(timeline_events, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(timeline_events)} timeline events")

    # Scrape Wikipedia engagements with dates
    if args.source in ['all', 'engagements']:
        engagements = wiki_scraper.scrape_engagements_with_dates()

        # Save engagements
        with open(args.output_dir / "wikipedia_engagements.json", 'w', encoding='utf-8') as f:
            json.dump(engagements, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(engagements)} engagements")

    # Consolidate
    if not args.skip_consolidate and (timeline_events or engagements):
        consolidator = TimelineConsolidator(args.output_dir)
        consolidated = consolidator.consolidate(timeline_events, engagements)
        consolidator.save(consolidated)

    # Print summary
    print("\n" + "="*60)
    print("TIMELINE SCRAPING COMPLETE")
    print("="*60)
    print(f"Timeline events: {len(timeline_events)}")
    print(f"Military engagements: {len(engagements)}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
