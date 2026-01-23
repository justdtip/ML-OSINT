#!/usr/bin/env python3
"""
ENTSO-E Ukraine Electricity Data Downloader via SFTP

Downloads bulk CSV data from ENTSO-E Transparency Platform SFTP server.
This method is suitable for large historical datasets.

NOTE: SFTP server planned to be discontinued by end of September 2025.
      Use File Library (https://fms.tp.entsoe.eu) as the new alternative.

Requirements:
    - pip install paramiko pandas

SFTP Credentials (from ENTSO-E registration):
    - Host: sftp-transparency.entsoe.eu
    - Port: 22
    - Username: Your registered email
    - Password: Your account password (NOT API key)

    export ENTSOE_USERNAME="your.email@example.com"
    export ENTSOE_PASSWORD="your-password"

File Naming Convention:
    Files are organized by data item and time period.
    Example: {Year}_{Month}_{DataItem}_{AreaCode}.csv

    Key directories:
    - ActualGenerationOutput_16.1.B&C - Generation by type
    - ActualTotalLoad_6.1.A - Total load
    - PhysicalFlows_12.1.G - Cross-border flows

Usage:
    python download_entsoe_sftp.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("Please install paramiko: pip install paramiko")
    sys.exit(1)

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SFTP Configuration
SFTP_HOST = 'sftp-transparency.entsoe.eu'
SFTP_PORT = 22
OUTPUT_DIR = Path(__file__).parent

# Ukraine EIC Code
UKRAINE_AREA_CODE = '10Y1001C--00038X'

# Key data directories on SFTP (these are approximate - actual structure varies)
DATA_ITEMS = {
    'generation': 'ActualGenerationOutput',
    'load': 'ActualTotalLoad',
    'crossborder': 'PhysicalFlows',
    'capacity': 'InstalledCapacity',
    'forecast_load': 'DayAheadTotalLoadForecast',
    'forecast_gen': 'DayAheadGenerationForecast'
}


def get_sftp_credentials():
    """Get SFTP credentials from environment variables."""
    username = os.environ.get('ENTSOE_USERNAME')
    password = os.environ.get('ENTSOE_PASSWORD')

    if not username or not password:
        logger.warning("""
SFTP credentials not found in environment variables.

To set credentials:
    export ENTSOE_USERNAME="your.email@example.com"
    export ENTSOE_PASSWORD="your-password"

Note: These are your ENTSO-E Transparency Platform login credentials,
      NOT the API key. Register at https://transparency.entsoe.eu/
""")
        return None, None

    return username, password


def connect_sftp(username, password):
    """Establish SFTP connection."""
    try:
        transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        logger.info(f"Connected to {SFTP_HOST}")
        return sftp, transport
    except Exception as e:
        logger.error(f"SFTP connection failed: {e}")
        return None, None


def list_directory(sftp, path='/'):
    """List contents of a directory on SFTP server."""
    try:
        items = sftp.listdir_attr(path)
        return [(item.filename, item.st_size, item.st_mtime) for item in items]
    except Exception as e:
        logger.error(f"Failed to list {path}: {e}")
        return []


def find_ukraine_files(sftp, base_path='/', search_term='10Y1001C--00038X'):
    """Search for files containing Ukraine data."""
    ukraine_files = []

    def search_recursive(path, depth=0):
        if depth > 5:  # Limit recursion depth
            return

        items = list_directory(sftp, path)
        for name, size, mtime in items:
            full_path = f"{path}/{name}" if path != '/' else f"/{name}"

            if search_term in name or 'UA' in name.upper():
                ukraine_files.append((full_path, size, mtime))
                logger.info(f"Found: {full_path}")

            # If it's a directory, search inside
            if size == 0 or '.' not in name:  # Likely a directory
                try:
                    search_recursive(full_path, depth + 1)
                except:
                    pass

    search_recursive(base_path)
    return ukraine_files


def download_file(sftp, remote_path, local_path):
    """Download a single file from SFTP."""
    try:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, str(local_path))
        logger.info(f"Downloaded: {remote_path} -> {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {remote_path}: {e}")
        return False


def explore_sftp_structure(sftp):
    """Explore and document the SFTP directory structure."""
    logger.info("Exploring SFTP directory structure...")

    structure = {}
    root_items = list_directory(sftp, '/')

    for name, size, mtime in root_items[:20]:  # Limit to first 20 items
        structure[name] = {
            'size': size,
            'modified': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d') if mtime else 'N/A'
        }

        # Try to list subdirectories
        try:
            sub_items = list_directory(sftp, f'/{name}')
            structure[name]['contents'] = [s[0] for s in sub_items[:10]]
        except:
            structure[name]['contents'] = []

    return structure


def main():
    """Main function to explore and download Ukraine data from ENTSO-E SFTP."""
    logger.info("=" * 60)
    logger.info("ENTSO-E SFTP Bulk Data Downloader")
    logger.info("=" * 60)

    username, password = get_sftp_credentials()
    if not username:
        logger.error("Cannot proceed without credentials.")
        sys.exit(1)

    sftp, transport = connect_sftp(username, password)
    if not sftp:
        logger.error("Failed to establish SFTP connection.")
        sys.exit(1)

    try:
        # Explore directory structure
        structure = explore_sftp_structure(sftp)

        logger.info("\nSFTP Directory Structure:")
        for name, info in structure.items():
            logger.info(f"  /{name}")
            if info.get('contents'):
                for sub in info['contents'][:5]:
                    logger.info(f"    - {sub}")

        # Look for Ukraine files
        logger.info("\nSearching for Ukraine data files...")
        ukraine_files = find_ukraine_files(sftp)

        if ukraine_files:
            logger.info(f"\nFound {len(ukraine_files)} files with Ukraine data")

            # Download files
            for remote_path, size, mtime in ukraine_files:
                filename = Path(remote_path).name
                local_path = OUTPUT_DIR / 'sftp_data' / filename
                download_file(sftp, remote_path, local_path)
        else:
            logger.info("No Ukraine-specific files found in search.")
            logger.info("Try browsing the directory structure manually.")

    finally:
        if transport:
            transport.close()
            logger.info("SFTP connection closed.")


if __name__ == '__main__':
    main()
