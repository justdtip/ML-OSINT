#!/usr/bin/env python3
"""
ENTSO-E Ukraine Electricity Data - Web Download Script

Downloads publicly visible data from ENTSO-E Transparency Platform website.
This script provides direct URLs for manual download and attempts to fetch
available CSV exports from the platform.

NOTE: For full programmatic access, an API key is required.
      This script provides URLs and structure for manual download.

Requirements:
    pip install requests pandas beautifulsoup4

Ukraine Data Context:
    - Ukraine synchronized with ENTSO-E on February 24, 2022
    - Bidding Zone: UA (BZN) / IPS of Ukraine
    - EIC Code: 10Y1001C--00038X
    - Data available from March 2022 onwards

Data URLs:
    Generation: https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show
    Load: https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show
    Cross-border: https://transparency.entsoe.eu/transmission-domain/physicalFlow/show
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install requests pandas beautifulsoup4")
    sys.exit(1)

# Configuration
OUTPUT_DIR = Path(__file__).parent
BASE_URL = 'https://transparency.entsoe.eu'

# Ukraine identifiers
UKRAINE_BZN = 'CTY|10Y1001C--00038X!BZN|10Y1001C--00038X'
UKRAINE_CTA = 'CTY|10Y1001C--00038X!CTA|10YUA-WEPS-----0'

# Direct URLs for Ukraine data (browser access)
DIRECT_URLS = {
    'generation': f'{BASE_URL}/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime=01.03.2022+00:00|CET|DAY&dateTime.endDateTime=01.03.2022+00:00|CET|DAY&area.values={UKRAINE_BZN}&productionType.values=B01&productionType.values=B02&productionType.values=B03&productionType.values=B04&productionType.values=B05&productionType.values=B06&productionType.values=B09&productionType.values=B10&productionType.values=B11&productionType.values=B12&productionType.values=B13&productionType.values=B14&productionType.values=B15&productionType.values=B16&productionType.values=B17&productionType.values=B18&productionType.values=B19&productionType.values=B20&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)',
    'load': f'{BASE_URL}/load-domain/r2/totalLoadR2/show?name=&defaultValue=false&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime=01.03.2022+00:00|CET|DAY&dateTime.endDateTime=01.03.2022+00:00|CET|DAY&area.values={UKRAINE_BZN}&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)',
    'cross_border': f'{BASE_URL}/transmission-domain/physicalFlow/show?name=&defaultValue=false&viewType=TABLE&areaType=CTY&atch=false&dateTime.dateTime=01.03.2022+00:00|CET|DAY&dateTime.endDateTime=01.03.2022+00:00|CET|DAY&border.values=CTY|10Y1001C--00038X!CTY_CTY|10Y1001C--00038X_CTY_CTY|10YPL-AREA-----S&dateTime.timezone=CET_CEST'
}

# Production type codes
PRODUCTION_TYPES = {
    'B01': 'Biomass',
    'B02': 'Fossil Brown coal/Lignite',
    'B03': 'Fossil Coal-derived gas',
    'B04': 'Fossil Gas',
    'B05': 'Fossil Hard coal',
    'B06': 'Fossil Oil',
    'B07': 'Fossil Oil shale',
    'B08': 'Fossil Peat',
    'B09': 'Geothermal',
    'B10': 'Hydro Pumped Storage',
    'B11': 'Hydro Run-of-river and poundage',
    'B12': 'Hydro Water Reservoir',
    'B13': 'Marine',
    'B14': 'Nuclear',
    'B15': 'Other renewable',
    'B16': 'Solar',
    'B17': 'Waste',
    'B18': 'Wind Offshore',
    'B19': 'Wind Onshore',
    'B20': 'Other'
}

# Ukraine's cross-border interconnections
UKRAINE_BORDERS = {
    'UA-PL': 'Ukraine - Poland',
    'UA-SK': 'Ukraine - Slovakia',
    'UA-HU': 'Ukraine - Hungary',
    'UA-RO': 'Ukraine - Romania',
    'UA-MD': 'Ukraine - Moldova'
}


def create_api_request_template():
    """Generate template for API requests (requires API key)."""
    template = {
        'base_url': 'https://web-api.tp.entsoe.eu/api',
        'headers': {
            'Content-Type': 'application/xml',
            'SECURITY_TOKEN': '<YOUR_API_KEY>'
        },
        'example_requests': {
            'actual_generation': {
                'endpoint': '/api',
                'params': {
                    'securityToken': '<API_KEY>',
                    'documentType': 'A75',  # Actual generation per type
                    'processType': 'A16',   # Realised
                    'in_Domain': '10Y1001C--00038X',  # Ukraine
                    'periodStart': '202203010000',
                    'periodEnd': '202203020000'
                }
            },
            'total_load': {
                'endpoint': '/api',
                'params': {
                    'securityToken': '<API_KEY>',
                    'documentType': 'A65',  # System total load
                    'processType': 'A16',   # Realised
                    'outBiddingZone_Domain': '10Y1001C--00038X',
                    'periodStart': '202203010000',
                    'periodEnd': '202203020000'
                }
            },
            'physical_flows': {
                'endpoint': '/api',
                'params': {
                    'securityToken': '<API_KEY>',
                    'documentType': 'A11',  # Aggregated energy data report
                    'in_Domain': '10Y1001C--00038X',    # Ukraine (to)
                    'out_Domain': '10YPL-AREA-----S',   # Poland (from)
                    'periodStart': '202203010000',
                    'periodEnd': '202203020000'
                }
            }
        }
    }
    return template


def create_data_coverage_info():
    """Create information about available data coverage."""
    coverage = {
        'country': 'Ukraine',
        'bidding_zone': 'UA_IPS',
        'eic_code': '10Y1001C--00038X',
        'data_start': '2022-02-24',
        'entsoe_sync_date': '2022-02-24',
        'notes': [
            'Ukraine synchronized with ENTSO-E Continental Europe grid on Feb 24, 2022',
            'Before synchronization: Two separate trade zones existed',
            'Data reflects wartime conditions - expect gaps during infrastructure attacks',
            'Cross-border flows show emergency imports during power crises'
        ],
        'available_data_types': {
            'generation': {
                'description': 'Actual generation output by production type',
                'resolution': '60 minutes',
                'key_types': ['Nuclear', 'Thermal (Coal/Gas)', 'Hydro', 'Solar', 'Wind']
            },
            'load': {
                'description': 'Actual total electricity load/demand',
                'resolution': '60 minutes',
                'notes': 'Shows demand drops during blackouts'
            },
            'cross_border_flows': {
                'description': 'Physical electricity flows across borders',
                'resolution': '60 minutes',
                'borders': list(UKRAINE_BORDERS.values()),
                'notes': 'Emergency imports visible during attacks'
            },
            'installed_capacity': {
                'description': 'Installed generation capacity per type',
                'resolution': 'Annual',
                'notes': 'Shows capacity changes from war damage'
            }
        },
        'osint_value': {
            'infrastructure_status': 'Generation data shows which plants are operational',
            'blackout_detection': 'Load drops indicate power outages',
            'import_dependency': 'Cross-border flows show emergency imports',
            'damage_assessment': 'Capacity changes indicate infrastructure destruction'
        }
    }
    return coverage


def generate_download_instructions():
    """Generate step-by-step download instructions."""
    instructions = """
================================================================================
ENTSO-E UKRAINE DATA DOWNLOAD INSTRUCTIONS
================================================================================

METHOD 1: Web Interface (No Registration Required for Viewing)
--------------------------------------------------------------
1. Go to https://transparency.entsoe.eu/
2. Select data type:
   - Generation: Dashboard > Generation > Actual Generation per Production Type
   - Load: Dashboard > Load Domain > Actual Total Load
   - Cross-border: Dashboard > Transmission > Physical Flows

3. Set filters:
   - Area: UA (Ukraine IPS) or BZN|10Y1001C--00038X
   - Date range: From March 2022 onwards
   - View: Table view for CSV export

4. Click "Export Data" button (CSV format)

Note: Web exports are limited to short time periods. For bulk data, use API.


METHOD 2: REST API (Requires Free Registration + API Key)
----------------------------------------------------------
1. Register at https://transparency.entsoe.eu/
2. Send email to transparency@entsoe.eu with subject "Restful API access"
3. Get API key from Account Settings > Web API Security Token
4. Use the Python script: download_entsoe_ukraine.py

   export ENTSOE_API_KEY="your-token-here"
   python download_entsoe_ukraine.py


METHOD 3: SFTP Bulk Download (Requires Registration)
----------------------------------------------------
Note: SFTP being discontinued September 2025

1. Register at https://transparency.entsoe.eu/
2. Connect to sftp://sftp-transparency.entsoe.eu
   - Username: Your registered email
   - Password: Your account password
3. Navigate to data directories
4. Download CSV files for Ukraine (UA or 10Y1001C--00038X)


METHOD 4: File Library (New - Requires Registration)
-----------------------------------------------------
1. Register and login at https://transparency.entsoe.eu/
2. Access File Library at https://fms.tp.entsoe.eu
3. Navigate to TP_export folder
4. Download monthly CSV extracts


QUICK REFERENCE - Ukraine Identifiers
--------------------------------------
- Country Code: UA
- Bidding Zone: UA_IPS
- EIC Code: 10Y1001C--00038X
- Control Area: 10YUA-WEPS-----0
- Timezone: Europe/Kyiv (UTC+2/+3)


KEY DATA SERIES FOR OSINT ANALYSIS
-----------------------------------
1. Actual Generation by Type (16.1.B&C)
   - Shows nuclear, thermal, hydro, renewables output
   - Gaps indicate plant outages/damage

2. Actual Total Load (6.1.A)
   - System-wide electricity demand
   - Drops indicate blackouts/load shedding

3. Physical Cross-Border Flows (12.1.G)
   - Import/export volumes with EU neighbors
   - High imports = domestic generation shortfall

4. Installed Generation Capacity (14.1.A)
   - Changes show infrastructure damage/repair


NEIGHBORING COUNTRIES (for cross-border analysis)
-------------------------------------------------
- Poland: 10YPL-AREA-----S
- Slovakia: 10YSK-SEPS-----K
- Hungary: 10YHU-MAVIR----U
- Romania: 10YRO-TEL------P
- Moldova: 10Y1001A1001A990

================================================================================
"""
    return instructions


def save_reference_files():
    """Save reference files for data access."""
    # Save API template
    api_template = create_api_request_template()
    with open(OUTPUT_DIR / 'api_request_template.json', 'w') as f:
        json.dump(api_template, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'api_request_template.json'}")

    # Save coverage info
    coverage = create_data_coverage_info()
    with open(OUTPUT_DIR / 'ukraine_data_coverage.json', 'w') as f:
        json.dump(coverage, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'ukraine_data_coverage.json'}")

    # Save instructions
    instructions = generate_download_instructions()
    with open(OUTPUT_DIR / 'DOWNLOAD_INSTRUCTIONS.txt', 'w') as f:
        f.write(instructions)
    print(f"Saved: {OUTPUT_DIR / 'DOWNLOAD_INSTRUCTIONS.txt'}")

    # Save direct URLs
    with open(OUTPUT_DIR / 'direct_urls.json', 'w') as f:
        json.dump(DIRECT_URLS, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'direct_urls.json'}")


def main():
    """Main function."""
    print("=" * 60)
    print("ENTSO-E Ukraine Data - Reference Files Generator")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate and save reference files
    save_reference_files()

    # Print instructions
    print(generate_download_instructions())

    print("\nReference files have been created in:")
    print(f"  {OUTPUT_DIR}")
    print("\nTo download actual data, you need to:")
    print("1. Register at https://transparency.entsoe.eu/")
    print("2. Request API access via email to transparency@entsoe.eu")
    print("3. Set environment variable: export ENTSOE_API_KEY='your-key'")
    print("4. Run: python download_entsoe_ukraine.py")


if __name__ == '__main__':
    main()
