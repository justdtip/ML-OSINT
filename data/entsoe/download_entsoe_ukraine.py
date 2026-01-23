#!/usr/bin/env python3
"""
ENTSO-E Ukraine Electricity Data Downloader

Downloads electricity generation, load, and cross-border flow data for Ukraine (UA_IPS)
from the ENTSO-E Transparency Platform API.

Requirements:
    - pip install entsoe-py pandas

API Key Required:
    1. Register at https://transparency.entsoe.eu/
    2. Email transparency@entsoe.eu with subject "Restful API access"
    3. Get token from Account Settings > Web API Security Token
    4. Set environment variable: export ENTSOE_API_KEY="your-token-here"

Usage:
    python download_entsoe_ukraine.py

Ukraine Data Availability:
    - Ukraine synchronized with ENTSO-E on February 24, 2022
    - Full data available from March 2022 onwards
    - Country code: UA_IPS (EIC: 10Y1001C--00038X)

Data Categories Downloaded:
    1. Actual Generation by Type (hourly)
    2. Actual Load (hourly)
    3. Cross-Border Physical Flows (to/from neighbors)
    4. Installed Generation Capacity
    5. Day-Ahead Load Forecast
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('entsoe_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = Path(__file__).parent
COUNTRY_CODE = 'UA_IPS'  # Ukraine Integrated Power System
TIMEZONE = 'Europe/Kiev'

# Ukraine's neighboring countries for cross-border flows
UKRAINE_NEIGHBORS = {
    'PL': 'Poland',
    'SK': 'Slovakia',
    'HU': 'Hungary',
    'RO': 'Romania',
    'MD': 'Moldova'
}

# Date range - Ukraine joined ENTSO-E in late February 2022
START_DATE = '2022-03-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')


def get_api_key():
    """Get API key from environment variable or prompt user."""
    api_key = os.environ.get('ENTSOE_API_KEY')

    if not api_key:
        logger.warning("ENTSOE_API_KEY environment variable not set.")
        logger.info("""
To get an API key:
1. Register at https://transparency.entsoe.eu/
2. Email transparency@entsoe.eu with subject "Restful API access"
3. Get token from Account Settings > Web API Security Token
4. Run: export ENTSOE_API_KEY="your-token-here"
""")
        return None

    return api_key


def download_with_retry(func, *args, max_retries=3, delay=5, **kwargs):
    """Execute API call with retry logic for rate limiting."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except NoMatchingDataError:
            logger.warning(f"No data available for query")
            return None
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                wait_time = delay * (2 ** attempt)
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                logger.warning(f"Error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise
    return None


def download_in_chunks(client, query_func, start_date, end_date, chunk_months=3, **kwargs):
    """Download data in monthly chunks to avoid API limits."""
    all_data = []
    current_start = pd.Timestamp(start_date, tz=TIMEZONE)
    end = pd.Timestamp(end_date, tz=TIMEZONE)

    while current_start < end:
        current_end = min(current_start + pd.DateOffset(months=chunk_months), end)

        logger.info(f"  Downloading {current_start.strftime('%Y-%m')} to {current_end.strftime('%Y-%m')}...")

        try:
            data = download_with_retry(
                query_func,
                start=current_start,
                end=current_end,
                **kwargs
            )

            if data is not None and len(data) > 0:
                all_data.append(data)
                logger.info(f"    Retrieved {len(data)} records")
            else:
                logger.warning(f"    No data for this period")

        except Exception as e:
            logger.error(f"    Error: {e}")

        # Rate limiting - be polite to the API
        time.sleep(1)
        current_start = current_end

    if all_data:
        return pd.concat(all_data)
    return None


def download_generation_data(client):
    """Download actual generation by type."""
    logger.info("Downloading actual generation by type...")

    data = download_in_chunks(
        client,
        client.query_generation,
        START_DATE,
        END_DATE,
        country_code=COUNTRY_CODE
    )

    if data is not None:
        output_file = OUTPUT_DIR / 'ukraine_generation.csv'
        data.to_csv(output_file)
        logger.info(f"Saved generation data to {output_file}")
        logger.info(f"  Date range: {data.index.min()} to {data.index.max()}")
        logger.info(f"  Columns: {list(data.columns)}")
        return data
    return None


def download_load_data(client):
    """Download actual electricity load/demand."""
    logger.info("Downloading actual load data...")

    data = download_in_chunks(
        client,
        client.query_load,
        START_DATE,
        END_DATE,
        country_code=COUNTRY_CODE
    )

    if data is not None:
        output_file = OUTPUT_DIR / 'ukraine_load.csv'
        data.to_csv(output_file)
        logger.info(f"Saved load data to {output_file}")
        logger.info(f"  Date range: {data.index.min()} to {data.index.max()}")
        return data
    return None


def download_load_forecast(client):
    """Download day-ahead load forecast."""
    logger.info("Downloading day-ahead load forecast...")

    data = download_in_chunks(
        client,
        client.query_load_forecast,
        START_DATE,
        END_DATE,
        country_code=COUNTRY_CODE
    )

    if data is not None:
        output_file = OUTPUT_DIR / 'ukraine_load_forecast.csv'
        data.to_csv(output_file)
        logger.info(f"Saved load forecast data to {output_file}")
        return data
    return None


def download_crossborder_flows(client):
    """Download cross-border physical flows with all neighbors."""
    logger.info("Downloading cross-border flows...")

    all_flows = {}

    for neighbor_code, neighbor_name in UKRAINE_NEIGHBORS.items():
        logger.info(f"  {COUNTRY_CODE} <-> {neighbor_code} ({neighbor_name})...")

        # Flows FROM Ukraine TO neighbor
        try:
            export_data = download_in_chunks(
                client,
                client.query_crossborder_flows,
                START_DATE,
                END_DATE,
                chunk_months=6,
                country_code_from=COUNTRY_CODE,
                country_code_to=neighbor_code
            )
            if export_data is not None:
                all_flows[f'export_to_{neighbor_code}'] = export_data
        except Exception as e:
            logger.warning(f"    Could not get exports to {neighbor_code}: {e}")

        time.sleep(0.5)

        # Flows FROM neighbor TO Ukraine
        try:
            import_data = download_in_chunks(
                client,
                client.query_crossborder_flows,
                START_DATE,
                END_DATE,
                chunk_months=6,
                country_code_from=neighbor_code,
                country_code_to=COUNTRY_CODE
            )
            if import_data is not None:
                all_flows[f'import_from_{neighbor_code}'] = import_data
        except Exception as e:
            logger.warning(f"    Could not get imports from {neighbor_code}: {e}")

        time.sleep(0.5)

    if all_flows:
        # Combine all flows into a single DataFrame
        flows_df = pd.DataFrame(all_flows)
        output_file = OUTPUT_DIR / 'ukraine_crossborder_flows.csv'
        flows_df.to_csv(output_file)
        logger.info(f"Saved cross-border flows to {output_file}")
        logger.info(f"  Flow types: {list(flows_df.columns)}")
        return flows_df
    return None


def download_installed_capacity(client):
    """Download installed generation capacity by type."""
    logger.info("Downloading installed generation capacity...")

    # For capacity, we query year by year
    all_data = []

    for year in range(2022, datetime.now().year + 1):
        start = pd.Timestamp(f'{year}-01-01', tz=TIMEZONE)
        end = pd.Timestamp(f'{year}-12-31', tz=TIMEZONE)

        if end > pd.Timestamp(datetime.now(), tz=TIMEZONE):
            end = pd.Timestamp(datetime.now(), tz=TIMEZONE)

        try:
            data = download_with_retry(
                client.query_installed_generation_capacity,
                country_code=COUNTRY_CODE,
                start=start,
                end=end
            )
            if data is not None:
                data['year'] = year
                all_data.append(data)
                logger.info(f"  Got capacity data for {year}")
        except Exception as e:
            logger.warning(f"  Could not get capacity for {year}: {e}")

        time.sleep(0.5)

    if all_data:
        capacity_df = pd.concat(all_data)
        output_file = OUTPUT_DIR / 'ukraine_installed_capacity.csv'
        capacity_df.to_csv(output_file)
        logger.info(f"Saved installed capacity to {output_file}")
        return capacity_df
    return None


def download_generation_forecast(client):
    """Download generation forecast (wind/solar)."""
    logger.info("Downloading wind and solar generation forecast...")

    data = download_in_chunks(
        client,
        client.query_wind_and_solar_forecast,
        START_DATE,
        END_DATE,
        country_code=COUNTRY_CODE
    )

    if data is not None:
        output_file = OUTPUT_DIR / 'ukraine_renewable_forecast.csv'
        data.to_csv(output_file)
        logger.info(f"Saved renewable forecast data to {output_file}")
        return data
    return None


def create_summary_report(results):
    """Create a summary report of downloaded data."""
    report_lines = [
        "=" * 60,
        "ENTSO-E Ukraine Electricity Data - Download Summary",
        "=" * 60,
        f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Country Code: {COUNTRY_CODE}",
        f"Requested Period: {START_DATE} to {END_DATE}",
        "",
        "Data Files Downloaded:",
        "-" * 40
    ]

    for name, data in results.items():
        if data is not None:
            if isinstance(data, pd.DataFrame):
                report_lines.append(f"\n{name}:")
                report_lines.append(f"  - Records: {len(data)}")
                report_lines.append(f"  - Date range: {data.index.min()} to {data.index.max()}")
                if hasattr(data, 'columns'):
                    report_lines.append(f"  - Columns: {list(data.columns)[:10]}...")
            elif isinstance(data, pd.Series):
                report_lines.append(f"\n{name}:")
                report_lines.append(f"  - Records: {len(data)}")
                report_lines.append(f"  - Date range: {data.index.min()} to {data.index.max()}")
        else:
            report_lines.append(f"\n{name}: No data available")

    report_lines.extend([
        "",
        "=" * 60,
        "Notes:",
        "- Ukraine synchronized with ENTSO-E on February 24, 2022",
        "- Data reflects the integrated Ukrainian power system",
        "- Cross-border flows show emergency imports during attacks",
        "- Generation data shows impact of infrastructure damage",
        "=" * 60
    ])

    report = "\n".join(report_lines)

    report_file = OUTPUT_DIR / 'download_summary.txt'
    with open(report_file, 'w') as f:
        f.write(report)

    logger.info(f"\nSummary report saved to {report_file}")
    print(report)

    return report


def main():
    """Main function to download all Ukraine electricity data."""
    logger.info("=" * 60)
    logger.info("ENTSO-E Ukraine Electricity Data Downloader")
    logger.info("=" * 60)

    # Get API key
    api_key = get_api_key()
    if not api_key:
        logger.error("Cannot proceed without API key.")
        logger.info("Set ENTSOE_API_KEY environment variable and try again.")
        sys.exit(1)

    # Initialize client
    client = EntsoePandasClient(api_key=api_key)
    logger.info(f"Initialized ENTSO-E client")
    logger.info(f"Target: Ukraine ({COUNTRY_CODE})")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Download all data types
    results = {}

    try:
        results['generation'] = download_generation_data(client)
    except Exception as e:
        logger.error(f"Generation download failed: {e}")
        results['generation'] = None

    try:
        results['load'] = download_load_data(client)
    except Exception as e:
        logger.error(f"Load download failed: {e}")
        results['load'] = None

    try:
        results['load_forecast'] = download_load_forecast(client)
    except Exception as e:
        logger.error(f"Load forecast download failed: {e}")
        results['load_forecast'] = None

    try:
        results['crossborder_flows'] = download_crossborder_flows(client)
    except Exception as e:
        logger.error(f"Cross-border flows download failed: {e}")
        results['crossborder_flows'] = None

    try:
        results['installed_capacity'] = download_installed_capacity(client)
    except Exception as e:
        logger.error(f"Installed capacity download failed: {e}")
        results['installed_capacity'] = None

    try:
        results['renewable_forecast'] = download_generation_forecast(client)
    except Exception as e:
        logger.error(f"Renewable forecast download failed: {e}")
        results['renewable_forecast'] = None

    # Create summary
    create_summary_report(results)

    logger.info("\nDownload complete!")

    return results


if __name__ == '__main__':
    main()
