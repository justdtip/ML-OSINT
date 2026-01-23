# ACLED Ukraine Conflict Data Download Guide

**Last Updated:** January 20, 2026
**Date Range:** February 1, 2022 - Present

---

## Overview

ACLED (Armed Conflict Location & Event Data) provides comprehensive event-level conflict data for Ukraine, including:
- Political violence events (battles, explosions/remote violence, violence against civilians)
- Demonstration events (protests, riots)
- Strategic developments
- Infrastructure attacks (energy, health, education, residential)

---

## 1. Registration (Required)

ACLED requires free registration to download data. Two options:

### Option A: myACLED Account (Recommended)
1. Visit: https://acleddata.com/register/
2. Create account with email verification
3. Access Data Export Tool and curated data files

### Option B: Developer/API Access
1. Visit: https://developer.acleddata.com/
2. Register for API credentials
3. Receive API key and use registered email for authentication

---

## 2. Download Methods

### Method 1: Ukraine Conflict Monitor (Curated Data File)

**Best for:** Quick access to pre-filtered Ukraine data

1. Go to: https://acleddata.com/monitor/ukraine-conflict-monitor
2. Scroll to "Download Data" section
3. Available files:
   - **Ukraine Data File**: All political violence, demonstrations, and strategic developments (2020-present)
   - **Infrastructure Attacks Data**: Events tagged with civilian infrastructure categories

**Coverage:** February 24, 2022 - present (weekly updates)

---

### Method 2: Data Export Tool (Custom Filters)

**Best for:** Custom date ranges and specific event types

1. Go to: https://acleddata.com/data-export-tool/
2. Log in with myACLED account
3. Apply filters:
   - **Country:** Ukraine
   - **Date Range:** 2022-02-01 to present
   - **Event Types:** Select as needed (Battles, Explosions/Remote violence, Violence against civilians, Protests, Riots, Strategic developments)
4. Select export format: CSV, Excel, or JSON
5. Download data

---

### Method 3: API Access (Programmatic)

**Best for:** Automated data pipelines, regular updates

#### Authentication

First, obtain an OAuth token:

```bash
# Get access token (valid 24 hours)
curl -X POST "https://acleddata.com/oauth/token" \
  -d "username=YOUR_EMAIL" \
  -d "password=YOUR_PASSWORD" \
  -d "grant_type=password" \
  -d "client_id=acled"
```

Response contains:
- `access_token`: Use for API calls (24-hour validity)
- `refresh_token`: Use to get new access token (14-day validity)

#### API Endpoint

**Base URL:** `https://acleddata.com/api/acled/read`

#### Example API Calls

```bash
# Download Ukraine data 2022-present (CSV format)
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  "https://acleddata.com/api/acled/read?_format=csv&country=Ukraine&event_date=2022-02-01|2026-01-20&event_date_where=BETWEEN&limit=50000" \
  -o acled_ukraine_2022_present.csv

# Download specific fields only
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  "https://acleddata.com/api/acled/read?_format=csv&country=Ukraine&year=2022|2026&year_where=BETWEEN&fields=event_id_cnty|event_date|event_type|sub_event_type|actor1|actor2|admin1|location|latitude|longitude|fatalities|notes" \
  -o acled_ukraine_filtered.csv

# Download specific event types (battles and explosions only)
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  "https://acleddata.com/api/acled/read?_format=csv&country=Ukraine&year=2022|2026&year_where=BETWEEN&event_type=Battles|Explosions/Remote%20violence" \
  -o acled_ukraine_battles_explosions.csv
```

#### Key API Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `_format` | Output format | `csv`, `json`, `xml` |
| `country` | Country filter | `Ukraine` |
| `year` | Year filter | `2022\|2026` |
| `year_where` | Year operator | `BETWEEN`, `>`, `<` |
| `event_date` | Date filter | `2022-02-01\|2026-01-20` |
| `event_date_where` | Date operator | `BETWEEN` |
| `event_type` | Event type filter | `Battles\|Explosions/Remote violence` |
| `region` | Region code | `12` (Europe) |
| `admin1` | First admin level | Oblast name |
| `fields` | Select specific columns | `event_id_cnty\|event_date\|fatalities` |
| `limit` | Max rows returned | `50000` (default: 5000) |

#### Available Fields

```
event_id_cnty, event_date, year, time_precision, disorder_type, event_type,
sub_event_type, actor1, assoc_actor_1, inter1, actor2, assoc_actor_2, inter2,
interaction, civilian_targeting, iso, region, country, admin1, admin2, admin3,
location, latitude, longitude, geo_precision, source, source_scale, notes,
fatalities, tags, timestamp
```

---

### Method 4: HDX Alternative (Aggregated Data - No Account Required)

**Best for:** Quick aggregated statistics without registration

HDX provides ACLED-derived aggregated data:

```bash
# Download aggregated conflict events from HDX (no auth required)
curl "https://hapi.humdata.org/api/v2/coordination-context/conflict-events?location_code=UKR&output_format=csv&limit=100000" \
  -o /Users/daniel.tipton/ML_OSINT/data/hdx/hdx_conflict_events.csv
```

Direct download (XLSX files):
- Visit: https://data.humdata.org/dataset/ukraine-acled-conflict-data
- Download aggregated files (political violence, civilian targeting, demonstrations)

**Note:** HDX provides aggregated counts by country-month/country-year, NOT event-level data. For full event-level data, use ACLED directly.

---

## 3. Python Download Script

```python
#!/usr/bin/env python3
"""
ACLED Ukraine Data Downloader
Requires: pip install requests pandas
"""

import requests
import pandas as pd
from datetime import datetime
import os

# Configuration
ACLED_EMAIL = "YOUR_EMAIL"
ACLED_PASSWORD = "YOUR_PASSWORD"
OUTPUT_DIR = "/Users/daniel.tipton/ML_OSINT/data/acled"

def get_access_token(email: str, password: str) -> str:
    """Get OAuth access token from ACLED API."""
    response = requests.post(
        "https://acleddata.com/oauth/token",
        data={
            "username": email,
            "password": password,
            "grant_type": "password",
            "client_id": "acled"
        }
    )
    response.raise_for_status()
    return response.json()["access_token"]

def download_ukraine_data(
    token: str,
    start_date: str = "2022-02-01",
    end_date: str = None,
    output_file: str = None
) -> pd.DataFrame:
    """Download ACLED Ukraine conflict data."""

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if output_file is None:
        output_file = os.path.join(
            OUTPUT_DIR,
            f"acled_ukraine_{start_date}_{end_date}.csv"
        )

    # Build API URL
    base_url = "https://acleddata.com/api/acled/read"
    params = {
        "_format": "csv",
        "country": "Ukraine",
        "event_date": f"{start_date}|{end_date}",
        "event_date_where": "BETWEEN",
        "limit": "100000"
    }

    # Make request
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()

    # Save to file
    with open(output_file, "w") as f:
        f.write(response.text)

    # Return as DataFrame
    df = pd.read_csv(output_file)
    print(f"Downloaded {len(df)} events to {output_file}")

    return df

if __name__ == "__main__":
    # Get token
    token = get_access_token(ACLED_EMAIL, ACLED_PASSWORD)

    # Download Ukraine data from Feb 2022 to present
    df = download_ukraine_data(
        token=token,
        start_date="2022-02-01"
    )

    # Print summary
    print(f"\nData Summary:")
    print(f"  Total events: {len(df)}")
    print(f"  Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    print(f"  Event types: {df['event_type'].value_counts().to_dict()}")
    print(f"  Total fatalities: {df['fatalities'].sum()}")
```

---

## 4. Data Fields Description

| Field | Description |
|-------|-------------|
| `event_id_cnty` | Unique event identifier |
| `event_date` | Date of event (YYYY-MM-DD) |
| `event_type` | Main event category |
| `sub_event_type` | Detailed event subcategory |
| `actor1` | Primary actor involved |
| `actor2` | Secondary actor (if applicable) |
| `admin1` | First administrative level (Oblast) |
| `admin2` | Second administrative level (Raion) |
| `location` | Specific location name |
| `latitude` | Event latitude |
| `longitude` | Event longitude |
| `fatalities` | Reported fatalities (conservative estimate) |
| `notes` | Event description |
| `tags` | Additional tags (infrastructure, civilian targeting) |

---

## 5. Event Type Categories

| Event Type | Sub-Event Types |
|------------|-----------------|
| **Battles** | Armed clash, Government regains territory, Non-state actor overtakes territory |
| **Explosions/Remote violence** | Shelling/artillery/missile attack, Air/drone strike, Suicide bomb, Grenade, Remote explosive/landmine/IED |
| **Violence against civilians** | Attack, Abduction/forced disappearance, Sexual violence |
| **Protests** | Peaceful protest, Protest with intervention, Excessive force against protesters |
| **Riots** | Violent demonstration, Mob violence |
| **Strategic developments** | Agreement, Arrests, Change to group/activity, Disrupted weapons use, Headquarters/base established, Looting/property destruction, Non-violent transfer of territory, Other |

---

## 6. Rate Limits and Best Practices

- **Default limit:** 5,000 events per request
- **Maximum limit:** Specify `limit=100000` for larger downloads
- **Token validity:** Access tokens expire after 24 hours
- **Pagination:** For very large datasets, use multiple requests with date ranges
- **Attribution:** Always cite ACLED in publications and outputs

---

## 7. Data Storage Location

```
/Users/daniel.tipton/ML_OSINT/data/acled/
├── ACLED_UKRAINE_DOWNLOAD_GUIDE.md  (this file)
├── acled_ukraine_2022_present.csv   (full dataset after download)
└── (additional filtered exports)
```

---

## 8. Additional Resources

- **ACLED Website:** https://acleddata.com/
- **Ukraine Conflict Monitor:** https://acleddata.com/monitor/ukraine-conflict-monitor
- **API Documentation:** https://acleddata.com/acled-api-documentation/
- **ACLED Codebook:** https://acleddata.com/resources/codebook/
- **Data Export Tool:** https://acleddata.com/data-export-tool/
- **HDX Alternative:** https://data.humdata.org/dataset/ukraine-acled-conflict-data

---

## 9. License and Attribution

ACLED data is free for academic use. Commercial use requires a license.

**Required attribution:**
> Armed Conflict Location & Event Data Project (ACLED); www.acleddata.com

**Terms of Use:** https://acleddata.com/terms-of-use/

---

*Document generated for ML_OSINT project*
