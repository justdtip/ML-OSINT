#!/usr/bin/env python3
"""
ISW Assessment Downloader

Downloads ISW (Institute for the Study of War) daily assessment reports from the
Wayback Machine. These reports contain narrative analysis of the Russia-Ukraine
conflict that can be used to provide contextual understanding for the HAN model.

The script:
1. Queries the Wayback Machine CDX API for all archived ISW assessment URLs
2. Deduplicates by date (keeping earliest successful capture per day)
3. Downloads the HTML content from archived snapshots
4. Extracts and saves the text content
5. Creates a metadata index for temporal alignment with model data

Usage:
    python download_isw_assessments.py [--output-dir DIR] [--max-concurrent N]
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote

import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('isw_download.log')
    ]
)
logger = logging.getLogger(__name__)

# Wayback Machine endpoints
WAYBACK_CDX_API = "https://web.archive.org/cdx/search/cdx"
WAYBACK_WEB_PREFIX = "https://web.archive.org/web"

# ISW URL patterns for daily assessments (prefix match, no wildcards)
ISW_URL_PATTERNS = [
    "understandingwar.org/backgrounder/russian-offensive",
]

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests to be respectful
MAX_RETRIES = 3
RETRY_DELAY = 5.0


class ISWDownloader:
    """Downloads and processes ISW assessment reports from the Wayback Machine."""

    def __init__(self, output_dir: Path, max_concurrent: int = 5):
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        self.html_dir = output_dir / "html"
        self.text_dir = output_dir / "text"
        self.metadata_file = output_dir / "assessment_metadata.json"

        # Create directories
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)

        # Session for async requests
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore: Optional[asyncio.Semaphore] = None

        # Metadata tracking
        self.metadata = {
            "download_started": datetime.now().isoformat(),
            "assessments": {}
        }

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(timeout=timeout)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def query_cdx_api(self, url_pattern: str) -> list[dict]:
        """Query Wayback Machine CDX API for archived URLs matching pattern."""
        params = {
            "url": url_pattern,
            "matchType": "prefix",
            "output": "json",
            "fl": "urlkey,timestamp,original,mimetype,statuscode,digest,length",
            "filter": "statuscode:200",
            "collapse": "digest",  # Dedupe by content hash (reduces ~330k to ~3k)
        }

        try:
            # Use longer timeout for large response
            timeout = aiohttp.ClientTimeout(total=300)
            async with self.session.get(WAYBACK_CDX_API, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 1:  # First row is header
                        headers = data[0]
                        entries = [dict(zip(headers, row)) for row in data[1:]]

                        # Filter to only actual assessment URLs (with date in path)
                        assessment_entries = []
                        for entry in entries:
                            url = entry.get('original', '')
                            # Must contain "assessment-" followed by a month name
                            if re.search(r'assessment-(?:january|february|march|april|may|june|july|august|september|october|november|december)', url.lower()):
                                assessment_entries.append(entry)

                        logger.info(f"Filtered {len(entries)} entries to {len(assessment_entries)} assessment URLs")
                        return assessment_entries
                else:
                    logger.warning(f"CDX API returned status {response.status} for {url_pattern}")
        except Exception as e:
            logger.error(f"Error querying CDX API for {url_pattern}: {e}")

        return []

    def parse_assessment_date(self, url: str, wayback_timestamp: str = None) -> Optional[datetime]:
        """Extract the assessment date from the URL.

        Args:
            url: The original ISW URL
            wayback_timestamp: Wayback Machine capture timestamp (YYYYMMDDhhmmss format)
                Used to infer year when URL doesn't include one.
        """
        # Patterns like:
        # russian-offensive-campaign-assessment-april-1-2023
        # russian-offensive-assessment-july-8-2023
        # russian-offensive-campaign-assessment-april-1 (2022, no year)

        url_lower = url.lower()

        # Month name to number mapping
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }

        # Try pattern: month-day-year (explicit year in URL)
        pattern = r'assessment-(\w+)-(\d+)-(\d{4})'
        match = re.search(pattern, url_lower)
        if match:
            month_name, day, year = match.groups()
            if month_name in months:
                try:
                    return datetime(int(year), months[month_name], int(day))
                except ValueError:
                    pass

        # Try pattern: month-day (no year in URL, infer from wayback timestamp)
        pattern = r'assessment-(\w+)-(\d+)(?:[^0-9]|$)'
        match = re.search(pattern, url_lower)
        if match:
            month_name, day = match.groups()
            if month_name in months:
                # Infer year from wayback timestamp or default to 2022
                year = 2022
                if wayback_timestamp and len(wayback_timestamp) >= 4:
                    year = int(wayback_timestamp[:4])
                try:
                    return datetime(year, months[month_name], int(day))
                except ValueError:
                    pass

        return None

    def deduplicate_by_date(self, entries: list[dict]) -> dict[str, dict]:
        """Keep only the earliest capture for each assessment date."""
        by_date = {}

        for entry in entries:
            original_url = entry.get('original', '')
            timestamp = entry.get('timestamp', '')

            # Parse the assessment date from URL (pass timestamp to infer year if needed)
            assessment_date = self.parse_assessment_date(original_url, timestamp)
            if not assessment_date:
                logger.debug(f"Could not parse date from URL: {original_url}")
                continue

            date_key = assessment_date.strftime('%Y-%m-%d')

            # Keep earliest capture or update if this is earlier
            if date_key not in by_date or timestamp < by_date[date_key]['timestamp']:
                by_date[date_key] = {
                    **entry,
                    'assessment_date': date_key,
                    'parsed_date': assessment_date
                }

        return by_date

    async def download_snapshot(self, entry: dict) -> Optional[str]:
        """Download a single snapshot from Wayback Machine."""
        timestamp = entry['timestamp']
        original_url = entry['original']
        wayback_url = f"{WAYBACK_WEB_PREFIX}/{timestamp}id_/{original_url}"

        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    await asyncio.sleep(REQUEST_DELAY)  # Rate limiting

                    async with self.session.get(wayback_url) as response:
                        if response.status == 200:
                            return await response.text()
                        elif response.status == 429:  # Rate limited
                            wait_time = RETRY_DELAY * (attempt + 1)
                            logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.warning(f"HTTP {response.status} for {wayback_url}")
                            return None

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout for {wayback_url}, attempt {attempt + 1}")
                except Exception as e:
                    logger.error(f"Error downloading {wayback_url}: {e}")

                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)

        return None

    def extract_text(self, html: str) -> dict:
        """Extract relevant text content from ISW assessment HTML."""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        result = {
            'title': '',
            'content': '',
            'key_takeaways': [],
            'sections': {}
        }

        # Try to find the title
        title_elem = soup.find('h1')
        if title_elem:
            result['title'] = title_elem.get_text(strip=True)

        # Extract content from paragraph tags in the body
        # ISW site structure varies over time, but <p> tags are consistent
        body = soup.body
        if body:
            text_parts = []
            for elem in body.find_all(['p', 'h2', 'h3', 'li']):
                text = elem.get_text(strip=True)
                # Filter out very short fragments and navigation/boilerplate
                if text and len(text) > 20:
                    # Skip common boilerplate patterns
                    lower_text = text.lower()
                    if any(skip in lower_text for skip in [
                        'click here', 'download the pdf', 'sign up',
                        'subscribe', 'follow us', 'share this',
                        'copyright', 'all rights reserved'
                    ]):
                        continue
                    text_parts.append(text)

            result['content'] = '\n\n'.join(text_parts)

            # Try to find key takeaways section
            for header in body.find_all(['h2', 'h3', 'strong']):
                header_text = header.get_text(strip=True).lower()
                if 'key takeaway' in header_text or 'key point' in header_text:
                    # Get following list items or paragraphs
                    next_elem = header.find_next_sibling()
                    while next_elem and next_elem.name in ['ul', 'ol', 'p']:
                        if next_elem.name in ['ul', 'ol']:
                            for li in next_elem.find_all('li'):
                                result['key_takeaways'].append(li.get_text(strip=True))
                        else:
                            result['key_takeaways'].append(next_elem.get_text(strip=True))
                        next_elem = next_elem.find_next_sibling()
                    break

        return result

    async def process_entry(self, entry: dict) -> bool:
        """Download and process a single assessment entry."""
        date_key = entry['assessment_date']

        # Check if already downloaded
        html_file = self.html_dir / f"{date_key}.html"
        text_file = self.text_dir / f"{date_key}.json"

        if html_file.exists() and text_file.exists():
            logger.debug(f"Already downloaded: {date_key}")
            return True

        # Download HTML
        html = await self.download_snapshot(entry)
        if not html:
            logger.error(f"Failed to download: {date_key}")
            return False

        # Save HTML
        html_file.write_text(html, encoding='utf-8')

        # Extract and save text
        extracted = self.extract_text(html)
        extracted['date'] = date_key
        extracted['original_url'] = entry['original']
        extracted['wayback_timestamp'] = entry['timestamp']

        text_file.write_text(json.dumps(extracted, indent=2, ensure_ascii=False), encoding='utf-8')

        # Update metadata
        self.metadata['assessments'][date_key] = {
            'original_url': entry['original'],
            'wayback_timestamp': entry['timestamp'],
            'title': extracted['title'],
            'content_length': len(extracted['content']),
            'key_takeaways_count': len(extracted['key_takeaways'])
        }

        logger.info(f"Downloaded: {date_key} - {extracted['title'][:50]}...")
        return True

    async def download_all(self) -> dict:
        """Main download workflow."""
        logger.info("Starting ISW assessment download")

        # Step 1: Query CDX API for all matching URLs
        logger.info("Querying Wayback Machine CDX API...")
        all_entries = []

        for pattern in ISW_URL_PATTERNS:
            logger.info(f"Querying pattern: {pattern}")
            entries = await self.query_cdx_api(pattern)
            logger.info(f"  Found {len(entries)} entries")
            all_entries.extend(entries)
            await asyncio.sleep(1)  # Rate limit between queries

        logger.info(f"Total CDX entries: {len(all_entries)}")

        # Step 2: Deduplicate by assessment date
        unique_by_date = self.deduplicate_by_date(all_entries)
        logger.info(f"Unique assessment dates: {len(unique_by_date)}")

        if not unique_by_date:
            logger.error("No valid assessment URLs found")
            return self.metadata

        # Sort by date
        sorted_entries = sorted(unique_by_date.values(), key=lambda x: x['assessment_date'])

        # Step 3: Download all assessments
        logger.info(f"Downloading {len(sorted_entries)} assessments...")

        success_count = 0
        failed_dates = []

        with tqdm(total=len(sorted_entries), desc="Downloading") as pbar:
            # Process in batches to manage concurrency
            for entry in sorted_entries:
                success = await self.process_entry(entry)
                if success:
                    success_count += 1
                else:
                    failed_dates.append(entry['assessment_date'])
                pbar.update(1)

        # Step 4: Save metadata
        self.metadata['download_completed'] = datetime.now().isoformat()
        self.metadata['total_requested'] = len(sorted_entries)
        self.metadata['total_downloaded'] = success_count
        self.metadata['failed_dates'] = failed_dates
        self.metadata['date_range'] = {
            'start': sorted_entries[0]['assessment_date'],
            'end': sorted_entries[-1]['assessment_date']
        }

        self.metadata_file.write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')

        logger.info(f"Download complete: {success_count}/{len(sorted_entries)} successful")
        logger.info(f"Metadata saved to: {self.metadata_file}")

        return self.metadata


async def main():
    parser = argparse.ArgumentParser(description="Download ISW assessments from Wayback Machine")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "isw_assessments",
        help="Output directory for downloaded files"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent downloads (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    async with ISWDownloader(args.output_dir, args.max_concurrent) as downloader:
        metadata = await downloader.download_all()

        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total assessments: {metadata.get('total_requested', 0)}")
        print(f"Successfully downloaded: {metadata.get('total_downloaded', 0)}")
        if metadata.get('date_range'):
            print(f"Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
        if metadata.get('failed_dates'):
            print(f"Failed dates: {len(metadata['failed_dates'])}")
        print(f"\nFiles saved to: {args.output_dir}")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
