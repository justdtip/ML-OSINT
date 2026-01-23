# Ukraine Conflict Data Acquisition

Raw data from four sources for ML/OSINT analysis of the Ukraine conflict. Data acquired on 2026-01-18.

## Directory Structure

```
/data
├── deepstate/
│   ├── daily/                              # 554 daily GeoJSON files
│   │   └── deepstatemap_data_YYYYMMDD.geojson
│   ├── snapshots/
│   │   └── API_DOCUMENTATION.json          # API structure documentation
│   ├── deepstate-map-data.geojson.gz       # Compressed historical archive (14MB)
│   └── file_list.json                      # Metadata and download URLs
├── firms/
│   └── ACQUISITION_GUIDE.json              # API docs and download instructions
├── ucdp/
│   └── ged_events.csv                      # 31,547 Ukraine conflict events
└── README.md
```

## Data Sources

### 1. DeepState Territorial Control (GitHub Archive)
**Source:** https://github.com/cyterat/deepstate-map-data

| Metric | Value |
|--------|-------|
| Files | 554 daily GeoJSON files |
| Date Range | 2024-07-08 → 2026-01-18 |
| Update Frequency | Daily at 03:00 UTC |
| Content | MultiPolygon of Russian-occupied territory |

**Schema:**
- Type: FeatureCollection
- Geometry: MultiPolygon (single feature per file)
- Properties: None (date encoded in filename)
- CRS: urn:ogc:def:crs:OGC:1.3:CRS84 (WGS84)

**Note:** GitHub archive starts July 2024, not Feb 2022. For earlier data, compressed archive contains all historical geometries with date field.

### 2. DeepState Event Markers (API)
**Source:** https://deepstatemap.live/api/

| Endpoint | Auth Required | Description |
|----------|---------------|-------------|
| /api/history/last | No | Current snapshot with all features |
| /api/history/ | Yes | List of available timestamps |
| /api/history/{timestamp} | Yes | Historical snapshot |

**Current Snapshot Structure (2026-01-17):**
- Total features: 522
- Polygons: 113 (territorial control areas)
- Points: 409 (military markers)

**Point Marker Categories:**
- `images/icon-2.png`: Attack direction arrows (82+)
- `images/icon-3.png`: Military units/battalions (256+)
- `images/icon-4.png`: Army-level units (15+)
- `images/icon-6.png`: Airfields (54+)
- `images/icon-1.png`: Moskva cruiser marker
- `images/icon-5.png`: Crimean Bridge

**Limitation:** Historical API access requires authentication. Only current snapshot available without credentials.

### 3. NASA FIRMS (Fire Information for Resource Management System)
**Source:** https://firms.modaps.eosdis.nasa.gov/

**Status:** Requires API key registration (free)

**Ukraine Bounding Box:** `22.0, 44.0, 40.5, 52.5` (west, south, east, north)

**Available Products:**
| Product | Resolution | Coverage |
|---------|------------|----------|
| VIIRS S-NPP | 375m | 2012-01-20 → present |
| VIIRS NOAA-20 | 375m | 2020-01-01 → present |
| VIIRS NOAA-21 | 375m | 2023 → present |
| MODIS C6.1 | 1km | 2000-11-01 → present |
| Landsat | 30m | 2022-06-20 → present |

**Acquisition Steps:**
1. Register at https://urs.earthdata.nasa.gov/users/new
2. Get MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key/
3. Use API: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_SP/22.0,44.0,40.5,52.5/10/`

**Output Columns:** latitude, longitude, bright_ti4, acq_date, acq_time, satellite, confidence, frp

### 4. UCDP GED (Georeferenced Event Dataset)
**Source:** https://ucdpapi.pcr.uu.se/api/gedevents/25.1

**Status:** ✅ Acquired

| Metric | Value |
|--------|-------|
| Total Events | 31,547 |
| Country | Ukraine (country_id=369) |
| Date Range | 2014 → 2024 |
| Events Post-Feb 2022 | 27,979 |
| Total Fatalities | 245,068 (best estimate) |
| Missing Coordinates | 0 |

**Key Columns:**
- `latitude`, `longitude`: Event location (all within Ukraine bounds)
- `date_start`, `date_end`: Event date range
- `best_est`, `low_est`, `high_est`: Fatality estimates
- `deaths_a`, `deaths_b`, `deaths_civilians`, `deaths_unknown`: Breakdown
- `side_a`, `side_b`: Conflict parties
- `type_of_violence`: 1=state-based, 2=non-state, 3=one-sided
- `where_description`: Location description
- `source_article`: News sources

**Coordinate Validation:**
- Latitude: 44.38° to 52.34° ✓
- Longitude: 22.72° to 39.85° ✓
- All coordinates within expected Ukraine bounds

## Validation Checklist

- [x] DeepState daily files span July 2024 → present (554 files, no gaps in available range)
- [x] DeepState compressed archive available (14MB, contains historical data with dates)
- [ ] DeepState API snapshots contain Point features (requires auth for historical)
- [x] Current DeepState snapshot has 409 Point markers (military units, airfields, etc.)
- [ ] FIRMS data loaded (requires API key registration)
- [x] UCDP events have latitude, longitude, date_start, best (fatality estimate) fields
- [x] UCDP coordinates within Ukraine bounds (44-52.5°N, 22-40.5°E)

## Known Limitations

1. **DeepState GitHub Archive:** Only starts July 2024, not Feb 2022
2. **DeepState API:** Historical access requires authentication
3. **FIRMS:** Requires free API key registration before data can be downloaded
4. **UCDP:** Data through 2024 only (2025+ may be in Candidate dataset)

## Next Steps for Complete Acquisition

1. **FIRMS:** Register for Earthdata account and obtain MAP_KEY
2. **DeepState API:** Investigate authentication options for historical snapshots
3. **UCDP Candidate:** Fetch version 25.0.11 for more recent events (2025+)

## File Sizes

```
deepstate/daily/           ~44MB (554 × ~79KB each)
deepstate/compressed       14MB
ucdp/ged_events.csv        22MB
```

---
*Data acquired: 2026-01-18*
*Purpose: ML/OSINT analysis - raw acquisition only, no transformation*
