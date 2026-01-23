# Wayback Machine Archives for Discontinued/Historical Data Sources

This document provides direct download links to archived versions of discontinued conflict monitoring data sources.

---

## 1. Google Community Mobility Reports (Discontinued October 15, 2022)

**Original URL:** https://www.google.com/covid19/mobility/
**Archive Status:** 5,445 snapshots available (April 3, 2020 - January 17, 2026)

### Direct Download Links (Latest Before Discontinuation)

**Global CSV (All Countries):**
```bash
# Last snapshot before discontinuation (October 2022)
curl -L -o google_mobility_global_20221015.csv \
  "https://web.archive.org/web/20221015120000/https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"

# Earlier snapshots (invasion period)
curl -L -o google_mobility_global_20220301.csv \
  "https://web.archive.org/web/20220301000000/https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"

curl -L -o google_mobility_global_20220401.csv \
  "https://web.archive.org/web/20220401000000/https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
```

**Ukraine-Specific PDF Reports:**
```bash
# Browse all Ukraine PDFs
# https://web.archive.org/web/*/https://www.gstatic.com/covid19/mobility/*Ukraine*

curl -L -o ukraine_mobility_20221015.pdf \
  "https://web.archive.org/web/20221015000000/https://www.gstatic.com/covid19/mobility/2022-10-15_UA_Mobility_Report_en.pdf"
```

**Alternative Archives:**
- HDX Archive: https://data.humdata.org/dataset/google-mobility-report
- GitHub Mirror: https://github.com/ActiveConclusion/COVID19_mobility

---

## 2. Apple Mobility Trends (Discontinued April 14, 2022)

**Original URL:** https://covid19.apple.com/mobility
**Archive Status:** Limited snapshots available

### Direct Download Links

```bash
# Wayback Machine
curl -L -o apple_mobility_wayback.csv \
  "https://web.archive.org/web/20220414000000/https://covid19-static.cdn-apple.com/covid19-mobility-data/current/applemobilitytrends.csv"

# GitHub Archive (recommended - complete dataset)
git clone https://github.com/ActiveConclusion/COVID19_mobility.git apple_mobility_archive
```

**Data Coverage:** January 13, 2020 - April 14, 2022
**Metrics:** Driving, transit, walking (relative to Jan 13, 2020 baseline)

---

## 3. RUSI Artillery Analysis Reports (PDF Only)

**Note:** RUSI provides analysis in PDF format only - no structured dataset exists.

### Archived Report URLs

```bash
# "Russia's Artillery War in Ukraine" (August 2023)
# Check Wayback for RUSI reports
curl "https://web.archive.org/cdx/search/cdx?url=rusi.org/*artillery*ukraine*&output=json" -o rusi_artillery_index.json

# "Winning the Industrial War" (2024)
# https://web.archive.org/web/*/https://rusi.org/*winning*industrial*
```

---

## 4. ISW Daily Assessments Archive

**Original URL:** https://www.understandingwar.org/
**Note:** ISW updates daily; Wayback captures historical assessments

### Finding Historical Assessments

```bash
# Get index of all ISW Ukraine updates
curl "https://web.archive.org/cdx/search/cdx?url=understandingwar.org/backgrounder/russian-offensive-campaign-assessment*&output=json&limit=1000" \
  -o isw_assessments_index.json

# Example: Specific date assessment
curl -L -o isw_assessment_20220301.html \
  "https://web.archive.org/web/20220301/https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-march-1"
```

---

## 5. NetBlocks Connectivity Reports

**URL:** https://netblocks.org/ukraine-crisis
**Status:** Active, but historical reports may be archived

```bash
# Check for historical snapshots
curl "https://web.archive.org/cdx/search/cdx?url=netblocks.org/*ukraine*&output=json&limit=500" \
  -o netblocks_ukraine_index.json
```

---

## 6. Copernicus EMS Activations Archive

**URL:** https://emergency.copernicus.eu/
**Ukraine Activations:** EMSR452, EMSN081, and ongoing

```bash
# Check archived activation pages
curl "https://web.archive.org/cdx/search/cdx?url=emergency.copernicus.eu/*EMSR452*&output=json" \
  -o copernicus_emsr452_index.json
```

---

## 7. UNHCR/IOM Historical Reports

**DTM Portal:** https://dtm.iom.int/ukraine
**Status:** Active with historical reports

```bash
# Get index of archived DTM reports
curl "https://web.archive.org/cdx/search/cdx?url=dtm.iom.int/ukraine*&output=json&limit=500" \
  -o iom_dtm_ukraine_index.json
```

---

## Batch Download Script for Wayback Archives

Save as `download_wayback_archives.sh`:

```bash
#!/bin/bash
# Download historical data from Wayback Machine
# Run from: /Users/daniel.tipton/ML_OSINT/data/wayback_archives/

set -e
WAYBACK_DIR="/Users/daniel.tipton/ML_OSINT/data/wayback_archives"
mkdir -p "$WAYBACK_DIR"/{google_mobility,apple_mobility,isw,netblocks}
cd "$WAYBACK_DIR"

echo "=== Downloading Google Mobility Archives ==="
cd google_mobility

# Key dates during Ukraine conflict
for date in 20220224 20220301 20220401 20220501 20220601 20220701 20220801 20220901 20221001 20221015; do
    echo "Downloading snapshot: $date"
    curl -L -f -o "global_mobility_${date}.csv" \
      "https://web.archive.org/web/${date}120000id_/https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv" 2>/dev/null || \
      echo "  Failed: $date"
    sleep 1  # Be nice to archive.org
done

echo "=== Downloading Apple Mobility Archive ==="
cd ../apple_mobility
curl -L -f -o "apple_mobility_final.csv" \
  "https://web.archive.org/web/20220414120000id_/https://covid19-static.cdn-apple.com/covid19-mobility-data/current/applemobilitytrends.csv" 2>/dev/null || \
  echo "Apple mobility download failed - try GitHub mirror"

echo "=== Getting CDX Indexes ==="
cd "$WAYBACK_DIR"

# ISW assessment index
curl -sf "https://web.archive.org/cdx/search/cdx?url=understandingwar.org/backgrounder/russian-offensive*&output=json&limit=2000" \
  -o isw_cdx_index.json

# NetBlocks Ukraine index
curl -sf "https://web.archive.org/cdx/search/cdx?url=netblocks.org/*ukraine*&output=json&limit=500" \
  -o netblocks_cdx_index.json

# Google Mobility index
curl -sf "https://web.archive.org/cdx/search/cdx?url=gstatic.com/covid19/mobility/Global_Mobility*&output=json&limit=2000" \
  -o google_mobility_cdx_index.json

echo "=== Complete ==="
echo "Downloaded to: $WAYBACK_DIR"
ls -la "$WAYBACK_DIR"
```

---

## Using the CDX API to Find Snapshots

The Wayback Machine's CDX API lets you discover all available snapshots:

```bash
# Basic query
curl "https://web.archive.org/cdx/search/cdx?url=DOMAIN/PATH&output=json"

# With filters
curl "https://web.archive.org/cdx/search/cdx?url=example.com/*&output=json&from=20220101&to=20221231&limit=100"

# Response format: [urlkey, timestamp, original, mimetype, statuscode, digest, length]
```

### Constructing Download URLs

From CDX results, construct download URLs:
```
https://web.archive.org/web/{timestamp}id_/{original_url}
```

The `id_` suffix returns the raw file without Wayback toolbar.

---

## Summary: Available Historical Data

| Source | Archive Status | Date Range | Download Size |
|--------|---------------|------------|---------------|
| Google Mobility CSV | ✅ 1,469 snapshots | Apr 2020 - Oct 2022 | ~200MB per snapshot |
| Google Mobility PDFs | ✅ Available | Apr 2020 - Oct 2022 | ~2MB per country |
| Apple Mobility | ✅ Available | Jan 2020 - Apr 2022 | ~15MB |
| ISW Assessments | ✅ ~2,000 pages | Feb 2022 - present | HTML |
| NetBlocks Reports | ✅ ~500 pages | Feb 2022 - present | HTML |

---

## Notes

1. **Rate Limiting:** Be respectful of archive.org - add 1-2 second delays between requests
2. **File Sizes:** Google Mobility CSVs are large (~200MB each) - download only needed dates
3. **Raw Downloads:** Use `id_` in URL to get raw file without Wayback modifications
4. **Alternative Sources:** Many discontinued sources have mirrors on GitHub, Kaggle, or HDX

---

*Document generated: January 20, 2026*
