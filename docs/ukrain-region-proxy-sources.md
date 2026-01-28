# OSINT data sources for Ukraine conflict tactical prediction at raion resolution

Regional-granularity conflict data exists across six key categories, with the **VIINA dataset**, **WarSpotting API**, and **DeepState territorial archives** forming the most ML-ready foundation for hierarchical attention networks. Most sources provide free API or bulk download access with historical coverage from February 2022, though regional personnel loss proxies remain the most challenging gap—requiring inference from Mediazona's Russian regional casualty breakdowns rather than direct Ukrainian raion-level data.

This report identifies **47 specific data sources** across your six requirement categories, prioritizing structured formats (GeoJSON, CSV, JSON APIs) with daily-or-better temporal resolution and raion/oblast spatial granularity. The recommended technical stack combines VIINA (6-hour village-level updates), WarSpotting (GPS-coordinate equipment losses), and DeepState GeoJSON archives for territorial dynamics—all free and API-accessible.

---

## Regional equipment loss proxies with coordinate-level precision

**WarSpotting** emerges as the highest-value source for regional equipment attrition, providing a public REST API with GPS coordinates for each documented loss. The API returns JSON with latitude/longitude, equipment type, status (destroyed/abandoned/captured), date, and unit attribution. Historical coverage extends to February 24, 2022, with daily updates and H3 hexagonal indexing available for spatial aggregation to raion boundaries. Russian losses are fully documented; Ukrainian losses have partial coverage.

**GeoConfirmed** supplements WarSpotting with KML exports containing geolocated events including equipment destruction. The dataset includes **5,455+ entries** with lat/lon coordinates, descriptions, source URLs, and timestamps. Direct download available at `cdn.geoconfirmed.org/geoconfirmed/kmls/ukraine-google-earth.kml`, with an additional ORBAT (order of battle) dataset tracking unit positions.

For satellite-based verification, three open data programs provide conflict-relevant imagery:
- **Capella Space Open Data**: Sub-0.25m SAR imagery via AWS S3 (cloud-penetrating, 24/7 coverage)
- **Umbra Space Open Data**: Industry-leading 16-25cm SAR with STAC API (`api.canopy.umbra.space/archive/search`)
- **Maxar Open Data Program**: 30-50cm GeoTIFF via AWS Registry for crisis activations (CC BY 4.0)

Copernicus Sentinel-2 (10m optical) and Sentinel-1 (5×20m SAR) provide free daily/5-day coverage through `dataspace.copernicus.eu`, suitable for change detection at equipment concentration sites but insufficient resolution for individual vehicle identification.

| Source | Format | Resolution | Update | Cost | API |
|--------|--------|------------|--------|------|-----|
| WarSpotting | JSON | GPS coords | Daily | Free | ✅ Public |
| GeoConfirmed | KML/API | GPS coords | Continuous | Free | ✅ |
| Capella Open Data | GeoTIFF/SICD | <0.25m | Archive | Free | AWS S3 |
| Umbra Open Data | GeoTIFF | 16-25cm | Weekly | Free | STAC API |
| Sentinel-1/2 | COG/GeoTIFF | 5-20m | Daily/5-day | Free | OData/STAC |

---

## Personnel loss proxies require Russian regional funeral data inference

Direct raion-level Ukrainian casualty data does not exist in open sources. However, **Mediazona/BBC Russian Service** maintains the most rigorous personnel loss database, documenting **155,000+ Russian deaths by name** with regional breakdowns by soldier's home oblast (Bashkortostan, Tatarstan, Sverdlovsk, etc.). Statistical modeling using Russian Probate Registry data estimates ~165,000 total Russian deaths as of February 2025. This provides a regional signal for Russian force degradation, though mapping to Ukrainian raions requires inferring deployment patterns.

Supplementary Russian casualty sources include:
- **topcargo200.com**: Fallen officers with biographies
- **poteru.net**: Soldier lists with unit information  
- **Google Sheets officer database**: `docs.google.com/spreadsheets/d/1_bpIqkzD88hlSpA-PDZenSQGNnVnxz3lwYHKViSyuUc`

For proxy inference of regional combat intensity correlating with casualties, **VIIRS nighttime lights** (`eogdata.mines.edu/products/vnl/`) at 500m resolution can detect population displacement and infrastructure damage patterns. Academic studies demonstrate VIIRS change detection correlates with conflict intensity in Ukrainian oblasts.

**IOM Displacement Tracking Matrix** provides the most granular population movement data at **raion (Admin 2) and hromada (Admin 3) levels**, with IDP/returnee estimates, demographic breakdowns, and origin oblast tracking. Data available via API (`dtm.iom.int/data-and-analysis/dtm-api`) and HDX downloads, with bi-monthly survey rounds since March 2022.

---

## Daily combat intensity beyond FIRMS thermal signatures

**VIINA (Violent Incident Information from News Articles)** represents the most ML-ready combat intensity dataset, providing automated updates every **6 hours** via GitHub. The dataset geocodes incidents to 33,141 populated places using GeoNames gazetteer, with BERT-classified event labels including `t_artillery`, `t_airstrike`, `t_uav`, `t_airalert`, `t_armor`, and `t_control`. Critical fields include ADM1_NAME, ADM2_NAME (raion), latitude/longitude, and territorial control status from DeepState, ISW, and Wikipedia (majority-vote combined).

For air activity, the **Ukraine Air Raid Alert API** (`alerts.com.ua/api/states`) provides real-time oblast-level alerts via REST and Server-Sent Events, with complete historical archives from March 2022. The **GitHub air raid sirens dataset** (`github.com/Vadimkin/ukrainian-air-raid-sirens-dataset`) offers daily CSV exports including alert start/end times by region—essential for training models on attack pattern prediction.

**ADS-B Exchange** provides military aircraft tracking with historical archives from March 2020 at 5-second intervals. The aircraft database (`downloads.adsbexchange.com/downloads/basic-ac-db.json.gz`) includes military designation flags. For specialized military tracking, **ADS-B.nl** filters and archives military transponder data from October 2016.

**ACLED** remains the academic standard for conflict events, providing weekly-updated geocoded incidents at village level with structured event typology (Battles, Explosions/Remote Violence, Violence Against Civilians). API access requires free registration; R package `acled.api` enables automated retrieval. HDX mirror provides aggregated raion-level analysis files.

| Source | Granularity | Update | Format | Access |
|--------|-------------|--------|--------|--------|
| VIINA | Village/ADM2 | 6 hours | CSV/GeoJSON | GitHub |
| Air Raid API | Oblast | Real-time | JSON/SSE | Free API key |
| ACLED | Village | Weekly | CSV/JSON | Registration |
| ADS-B Exchange | Aircraft position | Real-time | JSON | Free/commercial |
| UCDP GED | Village | Annual/monthly | CSV | Public |

---

## Logistics monitoring through rail infrastructure and territorial dynamics

**DeepState territorial control** provides the highest-fidelity frontline polygons, now accessible via daily GeoJSON archives maintained at `github.com/cyterat/deepstate-map-data`. The repository contains **400+ daily multipolygon snapshots** from February 2022, compressed as GeoJSON with occupation status properties. API access available at `deepstatemap.live/api/history/{timestamp}/geojson`.

**OpenRailwayMap** provides vector infrastructure data (tracks, stations, signals) for Russia, Ukraine, and Belarus via STAC-compliant API and OSM extracts. HDX mirrors the Ukraine railways shapefile with operator and type fields. For monitoring Russian military rail movements, **Conflict Intelligence Team** and **Project Hajun** track trainspotting reports from social media, though no structured API exists.

Satellite-based change detection uses:
- **Sentinel-1 SAR coherence time series**: Free via Google Earth Engine for damage proxy mapping
- **Copernicus Emergency Management Service (CEMS)**: Activation-based damage assessments with GeoJSON/Shapefile outputs for specific events

**UNOSAT damage assessments** provide building-level damage classification (destroyed/severe/moderate/possible) with GPS coordinates across **10,934+ documented entries** in 18+ AOIs. Data available on HDX in shapefile format—essential ground truth for training damage detection models.

For infrastructure status, **Open Infrastructure Map** (OpenStreetMap-based) provides critical infrastructure vectors (power, pipelines, telecoms), while **Eyes on Russia/Bellingcat** databases contain **23,000+ geolocated data points** on infrastructure damage, military movements, and civilian targeting.

---

## Information environment signals from Telegram and connectivity monitoring

**Telemetr.io** provides the most accessible Telegram analytics API with OpenAPI v3 documentation. The free tier offers 1,000 requests/month and 7-day history; paid tiers ($25-499/month) extend to 365-day archives for up to 10,000 channels. Key fields include subscriber growth, posting frequency, engagement trends, and citation networks between channels. API endpoint: `api.telemetr.io`.

**TGStat** offers broader coverage (historical data from 2017) with country/region filtering for Russia, Ukraine, and Belarus. However, US payment methods are not accepted; the R package `rtgstat` provides API access for researchers who can obtain credentials.

Internet connectivity monitoring serves as a powerful conflict proxy:
- **IODA** (`ioda.inetintel.cc.gatech.edu`): BGP visibility and traffic anomaly detection at AS level
- **Cloudflare Radar** (`radar.cloudflare.com/ua`): Real-time traffic trends and outage detection with Data Explorer for custom queries
- **NetBlocks**: Real-time ISP-level disruption alerts

For electricity infrastructure, **VIIRS Black Marble** products provide daily nighttime light radiance at 500m resolution, enabling detection of power outages across oblasts. Data available from `blackmarble.gsfc.nasa.gov` with complete archives from 2012. The IEA Ukraine Real-Time Electricity Data Explorer provides generation/consumption data for macro-level grid status.

Regional evacuation patterns available through:
- **IOM DTM**: Raion-level IDP estimates with origin tracking (bi-monthly)
- **UNHCR Operational Data Portal**: Oblast-level displacement with intentions surveys
- **HDX Ukraine 3W/5W**: Humanitarian organization activity by location

---

## Recommended data pipeline for hierarchical attention network training

The following architecture maximizes coverage across your six data categories while maintaining raion-level granularity and daily temporal resolution:

**Primary feature sources (free, API-accessible):**
1. **Combat events**: VIINA (6-hour village-level updates) + ACLED (weekly geocoded validation)
2. **Equipment losses**: WarSpotting API (GPS coordinates, daily)
3. **Territorial dynamics**: DeepState GeoJSON archives (daily multipolygons)
4. **Air activity**: Air raid alert API (real-time oblast) + VIINA t_uav/t_airalert flags
5. **Information environment**: Telemetr.io API (Telegram channel metrics)
6. **Infrastructure proxy**: VIIRS nighttime lights (daily radiance)

**Supplementary sources for model enrichment:**
- UCDP GED for historical baseline (1989-2024)
- ViEWS forecasting datasets for pre-computed conflict probability features
- PRIO-GRID (0.5°×0.5° cells) for standardized spatial feature engineering
- IOM DTM for population displacement at raion level

**Format standardization:**
All primary sources output GeoJSON, CSV, or JSON API responses suitable for direct tensor conversion. DeepState multipolygons can define regional attention boundaries; VIINA's ADM2 codes enable direct raion-level aggregation; WarSpotting GPS coordinates aggregate via H3 hexagonal indexing to arbitrary administrative boundaries.

**Temporal alignment:**
- VIINA/DeepState: Daily alignment possible
- ACLED: Weekly batch (Monday releases)
- Air raid alerts: Sub-minute resolution requiring daily aggregation
- VIIRS: Daily composites with 1-2 day latency

**Critical gap**: Direct raion-level Ukrainian personnel losses do not exist in open sources. Recommended proxy approach: combine Mediazona Russian regional casualty data (for adversary attrition), IOM displacement flows (for population stress indicators), and VIINA casualty flags (`t_civcas`, `t_milcas`) with geographic aggregation.

---

## Conclusion: A multi-source fusion approach addresses granularity requirements

The Ukrainian conflict has generated an unprecedented volume of structured OSINT data suitable for ML training. **VIINA emerges as the single most valuable source**, combining automated 6-hour updates, village-level geocoding, BERT-classified event types, and integrated territorial control from multiple map sources—all via free GitHub download with complete February 2022 coverage.

For equipment losses specifically, **WarSpotting's public API with GPS coordinates** fills the regional granularity gap that national-level aggregators like Oryx cannot provide. Combined with DeepState's daily territorial GeoJSON archives and the air raid alert API's real-time oblast coverage, these sources provide a strong foundation for hierarchical attention network training.

The primary remaining challenge is personnel loss attribution at raion level, which requires inference from Russian regional funeral data and displacement proxies rather than direct measurement. For commercial imagery augmentation, Capella and Umbra open data programs provide the best SAR coverage at no cost, while Planet Labs' academic program offers 3,000 km²/month of 3m optical imagery for research applications.