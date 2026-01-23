"""
Telegram scraping and text-image attribution for OSINT analysis.

This module provides tools to:
1. Scrape messages from Telegram channels using Telethon
2. Correctly attribute text to images (handling albums, replies, temporal proximity)
3. Download and deduplicate media
4. Prepare data for downstream LLM extraction and GNN ingestion
"""

from .models import ExtractedMessage, ScrapedChannel
from .scraper import TelegramScraper
from .attribution import TextImageAttributor
from .pipeline import IngestionPipeline

__all__ = [
    'ExtractedMessage',
    'ScrapedChannel',
    'TelegramScraper',
    'TextImageAttributor',
    'IngestionPipeline'
]
