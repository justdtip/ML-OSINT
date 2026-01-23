# VIINA 2.0 Dataset - Download Summary

**Downloaded:** 2026-01-20  
**Source:** https://github.com/zhukovyuri/VIINA  
**License:** Open Database License (ODbL)

## Overview

VIINA (Violent Incident Information from News Articles) 2.0 is a near-real time multi-source event data system for the 2022 Russian Invasion of Ukraine. The data are based on news reports from Ukrainian and Russian media, which were geocoded and classified into standard conflict event categories through machine learning (BERT).

## Date Range Coverage

| Year | Events | Date Range |
|------|--------|------------|
| 2022 | 259,851 | 2022-02-24 to 2022-12-31 |
| 2023 | 141,593 | 2023-01-01 to 2023-12-31 |
| 2024 | 97,885 | 2024-01-01 to 2024-12-31 |
| 2025 | 50,410 | 2025-01-01 to 2025-12-31 |
| 2026 | 1,833 | 2026-01-01 to 2026-01-19 |

**Total Events:** ~551,572

## File Structure

### Event-Level Data (extracted/)

1. **event_info_latest_YYYY.csv** - Raw event reports
   - Locations, dates, times, source URLs, news headlines
   - Columns: viina_version, event_id, event_id_1pd, date, time, geonameid, feature_code, asciiname, ADM1_NAME, ADM1_CODE, ADM2_NAME, ADM2_CODE, longitude, latitude, GEO_PRECISION, GEO_API, location, address, report_id, source, url, text, lang

2. **event_labels_latest_YYYY.csv** - Classified events (BERT model)
   - Actor and tactic labels for each event
   - 55+ classification columns including:
     - t_mil (military operations)
     - a_rus / a_ukr (Russian/Ukrainian forces involved)
     - a_rus_init / a_ukr_init (initiator)
     - t_airstrike, t_artillery, t_uav, t_armor, t_firefight, etc.

3. **event_1pd_latest_YYYY.csv** - De-duplicated events
   - "One-per-day" filtered to remove duplicate reports
   - Same structure as event_labels

### Territorial Control Data (extracted/)

4. **control_latest_YYYY.csv** - Daily control status (GeoNames locations)
   - N = 33,141 populated places
   - Columns: geonameid, date, status_wiki, status_boost, status_dsm, status_isw, status

5. **kontrol_latest_YYYY.csv** - Daily control status (KATOTTH locations)
   - N = 29,724 populated places
   - Uses Ukraine's national administrative register

### Geographic Data (extracted/)

6. **gn_UA_tess.geojson** - GeoNames tessellated geometries
7. **katotth_UA_tess.geojson** - KATOTTH tessellated geometries

## Data Sources

Ukrainian media:
- 24 Kanal (24tvua), Espreso TV, Forbes Ukraine, Interfax-Ukraine
- LIGA.net, LiveUAMap, Militarnyy, NV, Ukrainska Pravda, UNIAN

Russian media:
- Komsomolskaya Pravda, Mediazona, Meduza, Nezavisimaya Gazeta
- NTV, RIA Novosti

## Event Categories (Label Columns)

**Actors:**
- a_rus: Russian or Russian-aligned forces
- a_ukr: Ukrainian or Ukrainian-aligned forces
- a_rus_init/a_ukr_init: Event initiator
- a_civ: Civilian involvement
- a_other: Third party (US, EU, Red Cross, etc.)

**Tactics:**
- t_mil: War/military operations
- t_aad: Anti-air defense
- t_airstrike: Air strikes, bombing
- t_uav: Drones/UAVs
- t_artillery: Shelling, rockets, MLRS
- t_armor: Tank operations
- t_firefight: Small arms fire
- t_ied: Explosives, landmines
- t_raid: Special forces operations
- t_control: Territorial control claims
- t_property: Infrastructure destruction
- t_cyber: Cyber operations
- t_milcas/t_civcas: Military/civilian casualties

## Citation

Zhukov, Yuri and Natalie Ayers (2023). "VIINA 2.0: Violent Incident Information from News Articles on the 2022 Russian Invasion of Ukraine." Cambridge, MA: Harvard University.

## Repository Structure

```
/Users/daniel.tipton/ML_OSINT/data/viina/
├── extracted/                    # Extracted CSV and GeoJSON files
│   ├── event_info_latest_*.csv
│   ├── event_labels_latest_*.csv
│   ├── event_1pd_latest_*.csv
│   ├── control_latest_*.csv
│   ├── kontrol_latest_*.csv
│   ├── gn_UA_tess.geojson
│   └── katotth_UA_tess.geojson
├── VIINA-full/                   # Full git repository with LFS files
│   ├── Data/                     # Original zip archives
│   ├── Figures/                  # Maps and visualizations
│   ├── Diagnostics/              # Model diagnostics
│   └── README.md                 # Original documentation
└── VIINA_DATASET_README.md       # This file
```
