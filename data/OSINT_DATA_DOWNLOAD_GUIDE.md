# Ukraine Conflict Monitoring Data Sources - Download Guide

**Generated: January 20, 2026**

This guide provides download instructions for 27 conflict monitoring data sources. Sources are categorized by access requirements.

---

## Quick Reference: Access Requirements

| Category | Count | Sources |
|----------|-------|---------|
| **No Account Required** | 8 | VIINA, HDX, HDX HAPI, OSM, ReliefWeb, Maxar/Vantor, Ukraine Air Raid History, Ookla |
| **Free Account Required** | 10 | NASA Earthdata, ACLED, Copernicus Data Space, ERA5/CDS, ENTSO-E, Planet (limited), ASF DAAC, alerts.in.ua API, TGStat, Telemetr.io |
| **API Key/Token Required** | 5 | Liveuamap API, TomTom Traffic, Telegram API, Twitter/X API, alerts.in.ua (real-time) |
| **Commercial/Restricted** | 4 | Planet (full), MarineTraffic, Kpler, UN Global Platform AIS |

---

## Part 1: No Account Required - Direct Downloads

### 1.1 VIINA 2.0 Dataset (Georgetown University)

**Download entire repository:**
```bash
cd /Users/daniel.tipton/ML_OSINT/data/viina
curl -L -o viina_main.zip "https://github.com/zhukovyuri/VIINA/archive/refs/heads/main.zip"
unzip viina_main.zip
```

**Or download specific data files:**
```bash
# Control status data (ISW, DeepState, Wikipedia territorial control)
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/control_latest_2022.zip"
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/control_latest_2023.zip"
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/control_latest_2024.zip"
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/control_latest_2025.zip"
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/control_latest_2026.zip"

# Event data
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/event_1pd_latest_2022.zip"
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/event_1pd_latest_2023.zip"
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/event_1pd_latest_2024.zip"
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/event_1pd_latest_2025.zip"
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/event_1pd_latest_2026.zip"

# GeoJSON reference files
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/gn_UA_tess.geojson"
curl -L -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/katotth_UA_tess.geojson"
```

**Data format:** CSV in ZIP archives, GeoJSON
**Coverage:** February 24, 2022 - present (daily updates)
**License:** Open Database License (ODbL)

---

### 1.2 HDX (Humanitarian Data Exchange) Ukraine Datasets

**Browse all 246+ Ukraine datasets:**
https://data.humdata.org/dataset?q=ukraine

**Key datasets with direct download links:**

```bash
cd /Users/daniel.tipton/ML_OSINT/data/hdx

# Ukraine Admin Boundaries (COD)
curl -L -o ukr_admin_boundaries.zip "https://data.humdata.org/dataset/cod-ab-ukr/resource/download"

# Ukraine Population Statistics
curl -L -o ukr_population.csv "https://data.humdata.org/dataset/ukraine-population-statistics"

# IOM DTM Ukraine (displacement tracking)
# Visit: https://dtm.iom.int/ukraine for latest reports
```

**HDX HAPI (API Access - no authentication required):**
```bash
# Get Ukraine conflict events (ACLED integration)
curl "https://hapi.humdata.org/api/v2/coordination-context/conflict-events?location_code=UKR&output_format=csv" -o hdx_conflict_events.csv

# Get refugee data
curl "https://hapi.humdata.org/api/v2/affected-people/refugees-persons-of-concern?origin_location_code=UKR&output_format=csv" -o hdx_refugees.csv

# Get humanitarian needs
curl "https://hapi.humdata.org/api/v2/affected-people/humanitarian-needs?location_code=UKR&output_format=csv" -o hdx_humanitarian_needs.csv

# Get operational presence (3W data)
curl "https://hapi.humdata.org/api/v2/coordination-context/operational-presence?location_code=UKR&output_format=csv" -o hdx_operational_presence.csv
```

**Note:** For larger queries, add `app_identifier` parameter (base64 of "yourname:youremail")

---

### 1.3 OpenStreetMap Ukraine Data

**Full country extract:**
```bash
cd /Users/daniel.tipton/ML_OSINT/data/osm

# Download Ukraine PBF extract (Geofabrik - updated daily)
curl -L -O "https://download.geofabrik.de/europe/ukraine-latest.osm.pbf"

# Or from BBBike (alternative mirror)
curl -L -O "https://download.bbbike.org/osm/bbbike/Kiev/Kiev.osm.pbf"
```

**Overpass API queries (specific data extraction):**
```bash
# Get all power infrastructure in Ukraine
curl -X POST "https://overpass-api.de/api/interpreter" \
  -d 'data=[out:json][timeout:300];area["ISO3166-1"="UA"]->.ua;(node["power"](area.ua);way["power"](area.ua);relation["power"](area.ua););out body;>;out skel qt;' \
  -o ukraine_power_infrastructure.json

# Get all hospitals
curl -X POST "https://overpass-api.de/api/interpreter" \
  -d 'data=[out:json][timeout:300];area["ISO3166-1"="UA"]->.ua;(node["amenity"="hospital"](area.ua);way["amenity"="hospital"](area.ua););out body;>;out skel qt;' \
  -o ukraine_hospitals.json
```

**OSMCha (changeset analysis):**
API: https://osmcha.mapbox.com/api-docs/

---

### 1.4 ReliefWeb Ukraine Updates

```bash
cd /Users/daniel.tipton/ML_OSINT/data/reliefweb

# API access (no auth required, register appname)
curl "https://api.reliefweb.int/v1/reports?appname=ml_osint&filter[field]=country.name&filter[value]=Ukraine&limit=1000" -o reliefweb_reports.json

# Get disasters
curl "https://api.reliefweb.int/v1/disasters?appname=ml_osint&filter[field]=country.name&filter[value]=Ukraine" -o reliefweb_disasters.json
```

---

### 1.5 Maxar/Vantor Open Data Program

**Note:** Maxar Open Data has transitioned to Vantor. Access satellite imagery for humanitarian response.

**STAC Catalog:**
```bash
cd /Users/daniel.tipton/ML_OSINT/data/maxar

# Download catalog metadata
curl -L -o maxar_catalog.json "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"

# Browse visually: https://www.maxar.com/open-data (redirects to vantor.com/company/open-data-program/)
```

**Ukraine-specific events:** Search catalog for "Ukraine" events

---

### 1.6 Ookla Speedtest Open Data

```bash
cd /Users/daniel.tipton/ML_OSINT/data/ookla

# Download from AWS Open Data Registry
# Fixed broadband performance (quarterly tiles)
aws s3 cp --no-sign-request s3://ookla-open-data/parquet/performance/type=fixed/year=2024/quarter=4/ ./fixed_2024_q4/ --recursive

# Mobile performance
aws s3 cp --no-sign-request s3://ookla-open-data/parquet/performance/type=mobile/year=2024/quarter=4/ ./mobile_2024_q4/ --recursive
```

**Alternative (GitHub):**
```bash
git clone https://github.com/teamookla/ookla-open-data.git
```

**Format:** Apache GeoParquet, Shapefiles
**Resolution:** Zoom level 16 tiles (~610m)

---

### 1.7 Ukraine Air Raid Alert History

**Historical alerts (public endpoint):**
```bash
cd /Users/daniel.tipton/ML_OSINT/data/alerts_ua

# Get region list
curl "https://api.alerts.in.ua/v1/regions.json" -o regions.json

# Historical data requires API token - see Section 2
# Alternative: alerts.com.ua provides SSE streaming
curl "https://alerts.com.ua/api/states" -H "X-API-Key: YOUR_KEY" -o current_alerts.json
```

---

## Part 2: Free Account Required

### 2.1 NASA Earthdata (Black Marble VNP46A2)

**Registration:** https://urs.earthdata.nasa.gov/users/new

```bash
cd /Users/daniel.tipton/ML_OSINT/data/nasa_blackmarble

# After registration, create .netrc file:
echo "machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD" > ~/.netrc
chmod 600 ~/.netrc

# Download via LAADS DAAC (example - Ukraine tiles h19v03, h20v03, h20v04)
# Use the LAADS DAAC order system: https://ladsweb.modaps.eosdis.nasa.gov/search/order/1/VNP46A2--5000

# Or use Google Earth Engine:
# Asset: NASA/VIIRS/002/VNP46A2
```

**Python alternative:**
```python
import earthaccess
earthaccess.login()
results = earthaccess.search_data(
    short_name='VNP46A2',
    bounding_box=(22.1, 44.3, 40.2, 52.4),  # Ukraine bbox
    temporal=('2022-02-24', '2024-12-31')
)
earthaccess.download(results, '/Users/daniel.tipton/ML_OSINT/data/nasa_blackmarble/')
```

---

### 2.2 ACLED (Armed Conflict Location & Event Data)

**Registration:** https://developer.acleddata.com/

```bash
cd /Users/daniel.tipton/ML_OSINT/data/acled

# After getting API key and email verification:
curl "https://api.acleddata.com/acled/read?key=YOUR_KEY&email=YOUR_EMAIL&country=Ukraine&year=2022&year=2023&year=2024" -o acled_ukraine.json

# Export formats: JSON, CSV, XML
# Add &export_type=csv for CSV output
```

**Data includes:** Event type, sub_event_type, actors, fatalities, coordinates, infrastructure tags

---

### 2.3 Copernicus Data Space (Sentinel-1/2)

**Registration:** https://dataspace.copernicus.eu/

```bash
cd /Users/daniel.tipton/ML_OSINT/data/copernicus

# Get access token
TOKEN=$(curl -X POST "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token" \
  -d "grant_type=password&username=YOUR_EMAIL&password=YOUR_PASSWORD&client_id=cdse-public" | jq -r '.access_token')

# Search for Sentinel-2 over Ukraine
curl -H "Authorization: Bearer $TOKEN" \
  "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?\$filter=Collection/Name eq 'SENTINEL-2' and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((22.1 44.3, 40.2 44.3, 40.2 52.4, 22.1 52.4, 22.1 44.3))')&\$top=100" \
  -o sentinel2_search.json
```

**Alternative: Sentinel Hub API** or **Google Earth Engine**

---

### 2.4 ERA5 Weather Reanalysis (Copernicus CDS)

**Registration:** https://cds.climate.copernicus.eu/user/register

```python
# ~/.cdsapirc file:
# url: https://cds.climate.copernicus.eu/api
# key: YOUR_UID:YOUR_API_KEY

import cdsapi
c = cdsapi.Client()

c.retrieve('reanalysis-era5-single-levels', {
    'product_type': 'reanalysis',
    'variable': ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'total_precipitation'],
    'year': ['2022', '2023', '2024'],
    'month': [str(i).zfill(2) for i in range(1,13)],
    'day': [str(i).zfill(2) for i in range(1,32)],
    'time': ['00:00', '06:00', '12:00', '18:00'],
    'area': [52.4, 22.1, 44.3, 40.2],  # N, W, S, E
    'format': 'netcdf',
}, '/Users/daniel.tipton/ML_OSINT/data/era5/ukraine_era5_2022-2024.nc')
```

---

### 2.5 ENTSO-E Transparency Platform

**Registration:** https://transparency.entsoe.eu/ (free registration, email for API token)

```bash
cd /Users/daniel.tipton/ML_OSINT/data/entsoe

# Using entsoe-py library
pip install entsoe-py

# Python:
from entsoe import EntsoePandasClient
client = EntsoePandasClient(api_key='YOUR_TOKEN')
start = pd.Timestamp('2022-03-16', tz='Europe/Kiev')  # UA sync date
end = pd.Timestamp('2024-12-31', tz='Europe/Kiev')
df = client.query_load('UA_IPS', start=start, end=end)
df.to_csv('ukraine_load.csv')
```

---

### 2.6 ASF DAAC (Sentinel-1 SAR)

**Registration:** https://urs.earthdata.nasa.gov/

```bash
cd /Users/daniel.tipton/ML_OSINT/data/sentinel

# Using asf_search library
pip install asf_search

# Python:
import asf_search as asf
results = asf.geo_search(
    platform=[asf.PLATFORM.SENTINEL1],
    intersectsWith='POLYGON((22.1 44.3, 40.2 44.3, 40.2 52.4, 22.1 52.4, 22.1 44.3))',
    start='2022-02-24',
    end='2024-12-31',
    processingLevel=['GRD_HD']
)
# Download requires NASA Earthdata session
session = asf.ASFSession().auth_with_creds('username', 'password')
results.download(path='./', session=session)
```

---

### 2.7 Ukraine Air Raid Alert API (Real-time)

**Request token:** https://alerts.in.ua/api-request

```bash
# With token:
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "https://api.alerts.in.ua/v1/alerts/active.json" -o active_alerts.json

# Historical (month ago for specific region)
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "https://api.alerts.in.ua/v1/regions/9/alerts/month_ago.json" -o kyiv_month_history.json
```

**Rate limits:** 8-10 req/min soft, 12 req/min hard
**Python library:** `pip install alerts-in-ua`

---

## Part 3: Commercial/Restricted Access

### 3.1 Sources Requiring Commercial License

| Source | Access Method | Cost |
|--------|--------------|------|
| **Liveuamap API** | enterprise@liveuamap.com | Subscription |
| **Planet (Full)** | planet.com/contact | $10K+/month |
| **MarineTraffic AIS** | marinetraffic.com/api | Subscription + credits |
| **TomTom Traffic** | developer.tomtom.com | Free trial, then paid |
| **Twitter/X API** | $5,000/mo (Pro), $42,000/mo (Enterprise) | |

### 3.2 UN Global Platform AIS

**Academic access only:** Requires UNGP account approval for qualifying institutions.
Contact: unstats.un.org/wiki/display/AIS

---

## Part 4: Wayback Machine Archives

For discontinued sources or historical snapshots:

### 4.1 Google Mobility Reports (Discontinued Oct 2022)

```bash
cd /Users/daniel.tipton/ML_OSINT/data/wayback_archives

# Ukraine mobility reports archive
curl -L -o google_mobility_ua.csv "https://web.archive.org/web/20221015000000*/https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip"

# HDX archive of Google Mobility
curl -L "https://data.humdata.org/dataset/google-mobility-report" -o hdx_google_mobility_archive.html
```

### 4.2 Apple Mobility Trends (Discontinued April 2022)

```bash
# Archived on GitHub
git clone https://github.com/ActiveConclusion/COVID19_mobility.git
```

### 4.3 Check Any URL for Snapshots

```bash
# Wayback Machine API
curl "https://archive.org/wayback/available?url=URL_TO_CHECK" | jq

# CDX API for all snapshots
curl "https://web.archive.org/cdx/search/cdx?url=example.com&output=json" | jq
```

---

## Part 5: Download Script (All-in-One)

Save as `download_all.sh`:

```bash
#!/bin/bash
# Ukraine Conflict OSINT Data Downloader
# Run from: /Users/daniel.tipton/ML_OSINT/data/

set -e
BASE_DIR="/Users/daniel.tipton/ML_OSINT/data"

echo "=== Creating directories ==="
mkdir -p $BASE_DIR/{viina,hdx,osm,reliefweb,maxar,ookla,alerts_ua,wayback_archives}

echo "=== Downloading VIINA 2.0 ==="
cd $BASE_DIR/viina
curl -L -o viina_main.zip "https://github.com/zhukovyuri/VIINA/archive/refs/heads/main.zip"
unzip -o viina_main.zip

echo "=== Downloading HDX HAPI data ==="
cd $BASE_DIR/hdx
curl -s "https://hapi.humdata.org/api/v2/coordination-context/conflict-events?location_code=UKR&output_format=csv&limit=100000" -o hdx_conflict_events.csv
curl -s "https://hapi.humdata.org/api/v2/affected-people/refugees-persons-of-concern?origin_location_code=UKR&output_format=csv" -o hdx_refugees.csv

echo "=== Downloading OSM Ukraine ==="
cd $BASE_DIR/osm
curl -L -O "https://download.geofabrik.de/europe/ukraine-latest.osm.pbf"

echo "=== Downloading ReliefWeb data ==="
cd $BASE_DIR/reliefweb
curl -s "https://api.reliefweb.int/v1/reports?appname=ml_osint&filter[field]=country.name&filter[value]=Ukraine&limit=1000" -o reliefweb_reports.json

echo "=== Downloading Maxar catalog ==="
cd $BASE_DIR/maxar
curl -L -o maxar_catalog.json "https://maxar-opendata.s3.amazonaws.com/events/catalog.json" || echo "Maxar catalog unavailable"

echo "=== Downloading Ookla open data ==="
cd $BASE_DIR/ookla
git clone --depth 1 https://github.com/teamookla/ookla-open-data.git . 2>/dev/null || git pull

echo "=== Complete ==="
echo "Note: Sources requiring accounts (NASA, ACLED, Copernicus, etc.) must be downloaded separately after registration."
```

---

## Summary: What Can Be Downloaded Without Accounts

| Source | Size Estimate | Format |
|--------|--------------|--------|
| VIINA 2.0 | ~500MB | CSV, GeoJSON |
| HDX HAPI | ~50MB | CSV, JSON |
| OSM Ukraine | ~1.5GB | PBF |
| ReliefWeb | ~100MB | JSON |
| Maxar Catalog | ~5MB | JSON (metadata only) |
| Ookla Tiles | ~200MB/quarter | GeoParquet |

**Total immediate downloads:** ~2.5GB

---

## Contact/Support

- **VIINA:** yuri.zhukov@umich.edu
- **HDX:** hdx@un.org
- **ACLED:** admin@acleddata.com
- **Copernicus:** support@copernicus.eu
- **alerts.in.ua:** contact via website

---

*Document generated for ML_OSINT project. Last updated: January 20, 2026*
