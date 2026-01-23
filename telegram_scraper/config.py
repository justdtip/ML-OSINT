"""
Configuration management for Telegram scraper.

Handles:
- API credentials from environment variables / .env file
- Channel lists
- Output paths
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


@dataclass
class TelegramConfig:
    """Configuration for Telegram API access."""

    # Required: Get from https://my.telegram.org/apps
    api_id: int = 0
    api_hash: str = ""

    # Session file name (will be created on first run)
    session_name: str = "telegram_osint"

    # Output directories
    output_dir: Path = Path("./data/telegram")
    media_dir: Optional[Path] = None  # Defaults to output_dir/media

    # Scraping settings
    rate_limit_delay: float = 0.5  # Seconds between requests
    temporal_window_seconds: int = 60  # For temporal proximity attribution
    download_media: bool = True

    # Default channels to scrape (OSINT-focused)
    channels: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.media_dir is None:
            self.media_dir = self.output_dir / "media"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "TelegramConfig":
        """
        Load configuration from environment variables.

        Automatically loads .env file if python-dotenv is installed.

        Args:
            env_file: Optional path to .env file. If not specified, searches
                      current directory and parent directories.
        """
        # Load .env file if dotenv is available
        if HAS_DOTENV:
            if env_file:
                load_dotenv(env_file)
            else:
                # Search for .env in current and parent directories
                load_dotenv()  # Searches automatically

        api_id = os.environ.get("TELEGRAM_API_ID")
        api_hash = os.environ.get("TELEGRAM_API_HASH")

        if not api_id or not api_hash:
            raise ValueError(
                "TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables must be set.\n"
                "Get your credentials from: https://my.telegram.org/apps"
            )

        # Load default channels from channels.txt
        channels = cls._load_default_channels()

        return cls(
            api_id=int(api_id),
            api_hash=api_hash,
            session_name=os.environ.get("TELEGRAM_SESSION_NAME", "telegram_osint"),
            output_dir=Path(os.environ.get("TELEGRAM_OUTPUT_DIR", "./data/telegram")),
            download_media=os.environ.get("TELEGRAM_DOWNLOAD_MEDIA", "true").lower() == "true",
            channels=channels
        )

    @staticmethod
    def _load_default_channels() -> List[str]:
        """Load default channel list from channels.txt."""
        # Look for channels.txt in the same directory as this module
        module_dir = Path(__file__).parent
        channels_file = module_dir / "channels.txt"

        if not channels_file.exists():
            return []

        channels = []
        with open(channels_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    channels.append(line)

        return channels


def print_setup_instructions():
    """Print instructions for setting up Telegram API access."""
    print("""
================================================================================
TELEGRAM API SETUP INSTRUCTIONS
================================================================================

1. Go to https://my.telegram.org/apps

2. Log in with your phone number

3. Create a new application:
   - App title: OSINT Scraper (or any name)
   - Short name: osint_scraper (or any name)
   - Platform: Desktop
   - Description: Research tool

4. Copy your api_id and api_hash

5. Set environment variables:

   # Linux/Mac:
   export TELEGRAM_API_ID="your_api_id"
   export TELEGRAM_API_HASH="your_api_hash"

   # Or create a .env file:
   echo 'TELEGRAM_API_ID=your_api_id' >> .env
   echo 'TELEGRAM_API_HASH=your_api_hash' >> .env

6. Run the scraper:
   python -m telegram_scraper.cli --channel @channel_name

================================================================================
SECURITY NOTES
================================================================================

- NEVER commit your api_hash to version control
- Your personal Telegram account is used for scraping
- Aggressive scraping can get your account banned
- Only scrape public channels
- Respect rate limits (built into the scraper)

================================================================================
""")
