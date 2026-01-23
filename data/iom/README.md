# IOM DTM Ukraine Displacement Data

## Overview

This directory contains IOM (International Organization for Migration) Displacement Tracking Matrix (DTM) data for Ukraine, tracking internal displacement since February 2022.

**Data Source:** Humanitarian Data Exchange (HDX) - https://data.humdata.org/
**Original Provider:** IOM Displacement Tracking Matrix - https://dtm.iom.int/ukraine
**Download Date:** 2026-01-20

## Files

### 1. DTM API Time Series Data (CSV)
**File:** `ukr-iom-dtm-from-api-admin-0-to-1.csv`
- **Rows:** ~29,000
- **Date Coverage:** 2022-02-28 to 2025-11-30
- **Format:** CSV with HXL tags
- **Update Frequency:** Weekly

**Columns:**
| Column | Description |
|--------|-------------|
| id | Record identifier |
| operation | Operation name |
| admin0Name | Country name (Ukraine) |
| admin0Pcode | Country code (UKR) |
| admin1Name | Oblast name |
| admin1Pcode | Oblast P-code |
| admin2Name | Raion name (if applicable) |
| admin2Pcode | Raion P-code |
| adminLevel | Administrative level (0=country, 1=oblast) |
| numPresentIdpInd | Number of IDPs present |
| reportingDate | Date of report (YYYY-MM-DD) |
| yearReportingDate | Year |
| monthReportingDate | Month |
| roundNumber | Survey round number |
| displacementReason | Reason for displacement (Conflict) |
| numberMales | Male IDPs |
| numberFemales | Female IDPs |
| idpOriginAdmin1Name | Oblast of origin |
| idpOriginAdmin1Pcode | P-code of origin oblast |
| assessmentType | Type of assessment (BA=Baseline Assessment) |
| operationStatus | Status (Active) |

### 2. IDP Estimates (Excel Files)
**Files:**
- `idp-estimates-r20_publish.xlsx` - Round 20 (April 2025)
- `idp-estimates-r19_publish.xlsx` - Round 19 (December 2024)
- `idp-estimates-r18_publish.xlsx` - Round 18 (October 2024)
- `idp-estimates-r17_publish.xlsx` - Round 17 (August 2024)
- `idp-estimate-mar162022.xlsx` - Round 1 (March 2022 - baseline)

**Structure (per file):**
- Sheet 1: `Estimated Population` - IDP and returnee estimates by oblast
  - Round number, Country, Date, Oblast, Oblast pcode
  - Estimated de facto IDPs present
  - Estimated returnee population present
- Sheet 2: `Top 5 IDP hosting Oblasts` - Percentage distribution
- Sheet 3: `Top 5 Oblasts of origin of IDPs` - Origin breakdown
- Sheet 4: `Macro-regions_Oblasts` - Regional groupings

### 3. Returnees Data (Excel Files)
**Files:**
- `ukraine-returnees-dataset-round-20_hdx_publish.xlsx` - Round 20 (April 2025)
- `ukraine-returnees-dataset-round-19_hdx_publish.xlsx` - Round 19 (December 2024)
- `ukraine-returnees-dataset-round-18_hdx_publish.xlsx` - Round 18 (October 2024)

**Structure:**
- Sheet 1: `Returnees by Oblast` - Estimated returnee population by oblast
- Sheet 2: `Origin` - Top oblasts of origin for returnees
- Sheet 3: `Displacement` - Current displacement status of returnees

## Date Coverage Summary

| Dataset | Start Date | End Date | Rounds |
|---------|------------|----------|--------|
| API Time Series | Feb 2022 | Nov 2025 | 1-36+ |
| IDP Estimates | Mar 2022 | Apr 2025 | 1-20 |
| Returnees | Various | Apr 2025 | 18-20 |

## Key Statistics (Round 20, April 2025)

- **Estimated IDPs in Ukraine:** ~3.5 million
- **Estimated Returnees:** ~4.7 million
- **Top IDP-hosting oblasts:**
  - Dnipropetrovska (16%)
  - Kharkivska (12%)
  - Kyiv City (10%)
  - Kyivska Oblast (8%)
- **Top oblasts of origin:**
  - Donetska (29%)
  - Kharkivska (19%)
  - Khersonska (11%)
  - Zaporizka (10%)

## Methodology

IOM conducts General Population Surveys (GPS) through:
- Phone-based interviews with randomly selected respondents
- Typically 40,000+ screener interviews per round
- Follow-up interviews with ~1,400 IDPs, ~1,200 returnees, ~1,800 non-displaced

## Usage Notes

1. **P-codes:** Use Ukrainian administrative P-codes (UA##) for geographic matching
2. **Estimates:** Numbers for Donetska, Zaporizka, Luhanska, and Khersonska oblasts may be under-represented due to limited coverage of government-controlled areas
3. **HXL Tags:** CSV file includes HXL (Humanitarian Exchange Language) tags in row 2

## License

Source: International Organization for Migration (IOM), Displacement Tracking Matrix (DTM)
Materials may be viewed, downloaded, and printed for non-commercial use only.

## Additional Data (Requires Request)

The following datasets require data access requests through HDX:
- **Frontline Flow Monitoring & Population Baseline** - Settlement-level data within 25km of frontline
- **Baseline Assessment - Raion Level** - Registered IDP figures at raion/hromada level
- **Conditions of Return Assessment (CoRA)** - Detailed return conditions data
