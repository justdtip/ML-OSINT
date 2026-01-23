# HDX Ukraine Humanitarian Data Collection

## Overview

This directory contains humanitarian data for Ukraine downloaded from the Humanitarian Data Exchange (HDX) platform, focusing on the period from February 2022 (start of full-scale invasion) to present.

**Data Collection Date:** January 20, 2026
**Target Date Range:** 2022-02-01 to present
**Total Size:** ~69 MB

## Data Sources

### 1. HDX HAPI (Humanitarian API) v2

The primary data source is the HDX Humanitarian API (HAPI), which provides standardized humanitarian indicators from multiple sources.

**API Base URL:** `https://hapi.humdata.org/api/v2/`

**Authentication:** App identifier required (generated via `/api/v2/encode_app_identifier`)

### 2. HDX CKAN API

Additional datasets downloaded directly from HDX using the CKAN catalogue API.

**API Base URL:** `https://data.humdata.org/api/3/`

### 3. UNOSAT

Satellite-based building damage assessments from the United Nations Satellite Centre.

---

## Data Files

### HAPI API Data (CSV Format)

| File | Records | Size | Description |
|------|---------|------|-------------|
| `conflict_events_2022_present.csv` | 20,016 | 3.3 MB | ACLED conflict event data aggregated monthly by admin2 region |
| `food_prices_2022_present.csv` | 33,365 | 7.6 MB | WFP market price monitoring for food commodities |
| `humanitarian_needs_2022_present.csv` | 11,032 | 1.9 MB | OCHA humanitarian needs by sector and population status |
| `idps_2022_present.csv` | 1,118 | 208 KB | IOM Displacement Tracking Matrix - internally displaced persons |
| `refugees_from_ukraine_2022_present.csv` | 5,395 | 645 KB | UNHCR refugee and persons of concern data by country of asylum |
| `returnees_ukraine_2022_present.csv` | 78 | 9.3 KB | UNHCR returnee data |
| `rainfall_2022_present.csv` | 56,766 | 11 MB | Climate/rainfall data by region |
| `funding_2022_present.csv` | 14 | 2.3 KB | Humanitarian appeal funding (Flash Appeal, HRP) |
| `national_risk.csv` | 1 | 382 B | INFORM national risk assessment |
| `poverty_rate.csv` | 12 | 1.9 KB | Multidimensional poverty indicators |

### Additional Datasets

| File | Size | Description |
|------|------|-------------|
| `frontline_monitoring_082024_022025.xlsx` | 276 KB | AQLITY SLAC monitoring - settlement-level data on access, services, damage |

### UNOSAT Damage Assessments (Shapefiles)

Location: `unosat_damage_assessments/`

| File | Size | Description |
|------|------|-------------|
| `mariupol_livoberezhnyi_damage_may2022.zip` | 247 KB | Building damage assessment for Mariupol districts (May 2022) |
| `kharkiv_damage_assessment.zip` | 104 KB | Rapid damage assessment for North Kharkiv (March 2022) |
| `sumy_damage_assessment.zip` | 38 KB | Rapid damage assessment for Sumy (March 2022) |
| `kherson_damage_assessment.zip` | 44 MB | Damage assessment including Nova Kakhovka Dam flooding (June 2023) |

---

## Data Schema Details

### conflict_events_2022_present.csv

Fields from ACLED via HDX HAPI:
- `location_code`, `location_name` - ISO country code and name
- `admin1_code`, `admin1_name` - Oblast level
- `admin2_code`, `admin2_name` - Raion level
- `event_type` - Categories: `civilian_targeting`, `political_violence`, `demonstration`
- `events` - Count of events in period
- `fatalities` - Count of fatalities in period
- `reference_period_start`, `reference_period_end` - Monthly aggregation period

### idps_2022_present.csv

Fields from IOM Displacement Tracking Matrix:
- Geographic identifiers (location, admin1, admin2)
- `reporting_round` - DTM survey round number
- `assessment_type` - Assessment methodology
- `operation` - Type of displacement operation
- `population` - Estimated IDP population
- `reference_period_start`, `reference_period_end`

### humanitarian_needs_2022_present.csv

Fields from OCHA HPC Tools:
- Geographic identifiers
- `sector_code`, `sector_name` - Humanitarian sector (e.g., CCM, EDU, FSL, Health, Protection, Shelter, WASH)
- `category` - Population category (IDP, Non-Displaced, Returnee)
- `population_status` - `INN` (in need), `TGT` (targeted)
- `population` - Population count

### refugees_from_ukraine_2022_present.csv

Fields from UNHCR:
- `origin_location_code`, `origin_location_name` - Ukraine
- `asylum_location_code`, `asylum_location_name` - Country of asylum
- `population_group` - `REF` (refugee)
- `gender`, `age_range`, `min_age`, `max_age` - Demographics
- `population` - Population count
- `reference_period_start`, `reference_period_end` - Annual periods

---

## API Usage Examples

### Generate App Identifier

```bash
curl "https://hapi.humdata.org/api/v2/encode_app_identifier?application=my_app&email=my@email.com"
```

### Query Conflict Events

```bash
APP_ID="your_encoded_app_identifier"
curl "https://hapi.humdata.org/api/v2/coordination-context/conflict-events?location_code=UKR&start_date=2022-02-01&output_format=csv&limit=10000&app_identifier=${APP_ID}"
```

### Query IDPs

```bash
curl "https://hapi.humdata.org/api/v2/affected-people/idps?location_code=UKR&start_date=2022-02-01&output_format=csv&limit=10000&app_identifier=${APP_ID}"
```

### Query Humanitarian Needs

```bash
curl "https://hapi.humdata.org/api/v2/affected-people/humanitarian-needs?location_code=UKR&start_date=2022-02-01&output_format=csv&limit=10000&app_identifier=${APP_ID}"
```

### Check Data Availability

```bash
curl "https://hapi.humdata.org/api/v2/metadata/data-availability?location_code=UKR&output_format=json&app_identifier=${APP_ID}"
```

---

## Available Endpoints (HDX HAPI v2)

### Affected People
- `/api/v2/affected-people/idps` - Internally Displaced Persons (IOM DTM)
- `/api/v2/affected-people/refugees-persons-of-concern` - Refugees (UNHCR)
- `/api/v2/affected-people/returnees` - Returnees (UNHCR)
- `/api/v2/affected-people/humanitarian-needs` - Humanitarian needs (OCHA)

### Coordination Context
- `/api/v2/coordination-context/conflict-events` - Conflict events (ACLED)
- `/api/v2/coordination-context/funding` - Humanitarian funding (FTS)
- `/api/v2/coordination-context/national-risk` - Risk assessment (INFORM)
- `/api/v2/coordination-context/operational-presence` - 3W data (not available for Ukraine)

### Food Security & Nutrition
- `/api/v2/food-security-nutrition-poverty/food-security` - Food security (IPC)
- `/api/v2/food-security-nutrition-poverty/food-prices-market-monitor` - Market prices (WFP)
- `/api/v2/food-security-nutrition-poverty/poverty-rate` - Poverty (Oxford MPI)

### Climate
- `/api/v2/climate/rainfall` - Rainfall data

### Infrastructure
- `/api/v2/geography-infrastructure/baseline-population` - Population (not available for Ukraine)

---

## Data Limitations

1. **Baseline Population:** Not available through HAPI for Ukraine
2. **Food Security (IPC):** Not available through HAPI for Ukraine
3. **Operational Presence (3W):** Not available through HAPI for Ukraine
4. **API Rate Limits:** Max 10,000 records per request; pagination required for larger datasets
5. **Temporal Coverage:** Some datasets have gaps or delayed updates

---

## References

- [HDX Platform](https://data.humdata.org/)
- [HDX HAPI Documentation](https://hdx-hapi.readthedocs.io/)
- [HDX HAPI API Sandbox](https://hapi.humdata.org/docs)
- [HDX Developer Resources](https://data.humdata.org/faqs/devs)
- [UNOSAT Ukraine](https://unosat.org/products/)
- [HDX Ukraine Portal](https://data.humdata.org/group/ukr)

---

## Contact

HDX Support: hdx@un.org
