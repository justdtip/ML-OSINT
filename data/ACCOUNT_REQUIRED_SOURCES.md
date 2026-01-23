# Sources Requiring Account Registration

This document catalogues all conflict monitoring data sources that require account creation before data can be downloaded.

---

## Priority 1: Free Accounts with High-Value Data

### 1. NASA Earthdata (Black Marble Nighttime Lights)
- **URL:** https://urs.earthdata.nasa.gov/users/new
- **Product:** VNP46A2 - VIIRS Gap-Filled Lunar BRDF-Adjusted Nighttime Lights
- **Value:** Daily 500m resolution nighttime imagery for detecting power outages
- **Data Volume:** ~10GB per year for Ukraine tiles
- **Access Method:** After registration, use `earthaccess` Python library or LAADS DAAC web interface
- **Coverage:** January 2012 - present

### 2. ACLED (Armed Conflict Location & Event Data)
- **URL:** https://developer.acleddata.com/
- **Product:** Event-level conflict data with coordinates
- **Value:** Most comprehensive conflict event dataset, weekly updates
- **Data Volume:** ~50MB for Ukraine 2022-present
- **Access Method:** REST API with API key + registered email
- **Coverage:** January 2018 - present (intensified from Feb 2022)
- **Note:** Free for academic use (time-lagged); commercial license for real-time

### 3. Copernicus Data Space (Sentinel-1/2)
- **URL:** https://dataspace.copernicus.eu/
- **Products:**
  - Sentinel-1 SAR (all-weather change detection)
  - Sentinel-2 optical (NDVI vegetation monitoring)
- **Value:** Free high-resolution satellite imagery (10m)
- **Data Volume:** ~500GB+ for comprehensive Ukraine coverage
- **Access Method:** OData API, STAC, openEO, or Sentinel Hub
- **Coverage:** 2014 - present

### 4. Copernicus Climate Data Store (ERA5)
- **URL:** https://cds.climate.copernicus.eu/user/register
- **Product:** ERA5 hourly reanalysis data
- **Value:** Meteorological variables for fire weather analysis
- **Data Volume:** ~5GB per year for Ukraine bounding box
- **Access Method:** CDS API with Python `cdsapi` library
- **Coverage:** 1940 - present (~5 days latency)

### 5. ENTSO-E Transparency Platform
- **URL:** https://transparency.entsoe.eu/
- **Product:** European power grid data
- **Value:** Ukraine electricity generation/consumption since March 2022 sync
- **Data Volume:** ~100MB for Ukraine data
- **Access Method:** REST API (XML) or `entsoe-py` Python library
- **Coverage:** March 16, 2022 - present (Ukraine sync date)

### 6. ASF DAAC (Alaska Satellite Facility)
- **URL:** Uses NASA Earthdata login (same as #1)
- **Product:** Sentinel-1 SAR data with on-demand processing
- **Value:** Pre-processed RTC/InSAR products via HyP3
- **Access Method:** `asf_search` Python library
- **Note:** Same account as NASA Earthdata

---

## Priority 2: Free Accounts with Specialized Data

### 7. Ukraine Air Raid Alert API
- **URL:** https://alerts.in.ua/api-request
- **Product:** Real-time and historical air raid alerts
- **Value:** Oblast/raion/city level alert data since March 2022
- **Rate Limits:** 8-10 req/min soft, 12 req/min hard
- **Access Method:** Bearer token authentication
- **Python Library:** `pip install alerts-in-ua`

### 8. Telegram API (for OSINT channels)
- **URL:** https://my.telegram.org/apps
- **Products:** Full message history from public channels
- **Value:** Real-time conflict reporting from Ukrainian channels
- **Rate Limits:** ~5,000-6,000 posts/day
- **Access Method:** Telethon Python library with API ID/hash
- **Legal:** No ToS prohibition on public channel data collection

### 9. TGStat API
- **URL:** https://api.tgstat.ru
- **Product:** Telegram channel analytics (2.6M+ channels)
- **Value:** Channel statistics, mentions since 2017
- **Access Method:** Subscription tiers

### 10. Telemetr.io API
- **URL:** https://telemetr.io
- **Product:** Telegram analytics (1.8M+ channels, 65B+ messages)
- **Access Method:** REST API (OpenAPI v3)

---

## Priority 3: Commercial/Restricted (Catalogue Only)

### 11. Liveuamap API
- **Contact:** enterprise@liveuamap.com
- **Product:** Real-time geocoded conflict events
- **Cost:** Commercial subscription
- **Alternative:** RapidAPI (Maeplets Liveuamap API)

### 12. Planet Labs
- **URL:** https://www.planet.com/
- **Products:** Daily 3-5m imagery (PlanetScope)
- **Free Tier:**
  - Education & Research: 3,000 km²/month (30-day delay)
  - NASA CSDA: 5M km² quota (US federally-funded only)
- **Commercial:** $10K+/month

### 13. TomTom Traffic API
- **URL:** https://developer.tomtom.com/
- **Products:** Traffic Flow, Incidents, Stats
- **Free Tier:** 30-day trial via MOVE Portal
- **Commercial:** Pay-per-use after trial

### 14. MarineTraffic/Kpler (AIS Data)
- **URL:** https://www.marinetraffic.com/en/ais-api-services
- **Product:** Maritime vessel tracking
- **Cost:** Subscription + credits (1 credit/terrestrial, 10/satellite)

### 15. UN Global Platform AIS
- **URL:** https://unstats.un.org/wiki/display/AIS
- **Product:** Global AIS data (exactEarth/Orbcomm)
- **Access:** Academic institutions only, requires approval
- **Coverage:** October 2018 - present

### 16. Twitter/X API
- **Pricing (as of 2024):**
  - Basic: $100/month
  - Pro: $5,000/month
  - Enterprise: $42,000/month
- **Academic Access:** EU DSA-mandated, limited approvals

---

## Registration Checklist

```
[ ] 1. NASA Earthdata     - https://urs.earthdata.nasa.gov/users/new
[ ] 2. ACLED              - https://developer.acleddata.com/
[ ] 3. Copernicus Data    - https://dataspace.copernicus.eu/
[ ] 4. Copernicus CDS     - https://cds.climate.copernicus.eu/user/register
[ ] 5. ENTSO-E            - https://transparency.entsoe.eu/
[ ] 6. alerts.in.ua       - https://alerts.in.ua/api-request
[ ] 7. Telegram           - https://my.telegram.org/apps
```

---

## Post-Registration Download Commands

After registering for each service, use these commands:

### NASA Earthdata / ASF DAAC
```bash
# Create ~/.netrc
echo "machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD" > ~/.netrc
chmod 600 ~/.netrc

# Python download
import earthaccess
earthaccess.login()
results = earthaccess.search_data(short_name='VNP46A2', bounding_box=(22.1,44.3,40.2,52.4))
earthaccess.download(results, './nasa_blackmarble/')
```

### ACLED
```bash
curl "https://api.acleddata.com/acled/read?key=YOUR_KEY&email=YOUR_EMAIL&country=Ukraine&year=2022:2024&export_type=csv" -o acled_ukraine.csv
```

### Copernicus Data Space
```bash
# Get token
TOKEN=$(curl -X POST "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token" \
  -d "grant_type=password&username=EMAIL&password=PASSWORD&client_id=cdse-public" | jq -r '.access_token')

# Search
curl -H "Authorization: Bearer $TOKEN" "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?..."
```

### ERA5
```python
# ~/.cdsapirc: url: https://cds.climate.copernicus.eu/api\nkey: UID:API_KEY
import cdsapi
c = cdsapi.Client()
c.retrieve('reanalysis-era5-single-levels', {...}, 'output.nc')
```

### ENTSO-E
```python
# pip install entsoe-py
from entsoe import EntsoePandasClient
client = EntsoePandasClient(api_key='YOUR_TOKEN')
df = client.query_load('UA_IPS', start=start, end=end)
```

### Air Raid Alerts
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" "https://api.alerts.in.ua/v1/alerts/active.json"
```

---

*Document generated: January 20, 2026*
