# ENTSO-E Ukraine Electricity Data

This directory contains scripts and documentation for downloading electricity data for Ukraine (UA_IPS) from the ENTSO-E Transparency Platform.

## Background

Ukraine synchronized with the ENTSO-E Continental European grid on **February 24, 2022** - the same day Russia's full-scale invasion began. This emergency synchronization (originally planned for 2023) was completed within weeks, allowing Ukraine to import electricity from EU neighbors during power crises.

## Data Available

| Data Type | Description | Resolution | OSINT Value |
|-----------|-------------|------------|-------------|
| **Generation by Type** | Nuclear, thermal, hydro, solar, wind output | Hourly | Shows which plants are operational |
| **Total Load** | System-wide electricity demand | Hourly | Drops indicate blackouts |
| **Cross-Border Flows** | Import/export with EU neighbors | Hourly | Emergency imports during attacks |
| **Installed Capacity** | Generation capacity per type | Annual | Infrastructure damage assessment |

## Ukraine Identifiers

- **Country Code**: `UA`
- **Bidding Zone**: `UA_IPS`
- **EIC Code**: `10Y1001C--00038X`
- **Control Area**: `10YUA-WEPS-----0`
- **Timezone**: `Europe/Kyiv`

## Neighboring Countries

| Country | EIC Code | Border |
|---------|----------|--------|
| Poland | 10YPL-AREA-----S | UA-PL |
| Slovakia | 10YSK-SEPS-----K | UA-SK |
| Hungary | 10YHU-MAVIR----U | UA-HU |
| Romania | 10YRO-TEL------P | UA-RO |
| Moldova | 10Y1001A1001A990 | UA-MD |

## Scripts

### 1. `download_entsoe_ukraine.py` - API Download (Recommended)

Downloads data via the ENTSO-E REST API using the `entsoe-py` Python package.

**Requirements:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install entsoe-py pandas
```

**API Key Setup:**
1. Register at https://transparency.entsoe.eu/
2. Email `transparency@entsoe.eu` with subject "Restful API access"
3. Get token from Account Settings > Web API Security Token

```bash
export ENTSOE_API_KEY="your-token-here"
python download_entsoe_ukraine.py
```

**Output Files:**
- `ukraine_generation.csv` - Generation by type
- `ukraine_load.csv` - Total load
- `ukraine_crossborder_flows.csv` - Cross-border flows
- `ukraine_installed_capacity.csv` - Installed capacity
- `ukraine_load_forecast.csv` - Day-ahead load forecast
- `ukraine_renewable_forecast.csv` - Wind/solar forecast

### 2. `download_entsoe_sftp.py` - SFTP Bulk Download

Downloads bulk CSV files from ENTSO-E SFTP server.

**Note:** SFTP server will be discontinued by September 2025.

```bash
pip install paramiko pandas
export ENTSOE_USERNAME="your.email@example.com"
export ENTSOE_PASSWORD="your-password"
python download_entsoe_sftp.py
```

### 3. `download_entsoe_web.py` - Reference Generator

Generates reference files and download instructions.

```bash
python download_entsoe_web.py
```

## Manual Download

Visit https://transparency.entsoe.eu/ and navigate to:

- **Generation**: Dashboard > Generation > Actual Generation per Production Type
- **Load**: Dashboard > Load Domain > Actual Total Load
- **Cross-border**: Dashboard > Transmission > Physical Flows

Select:
- Area: UA (Ukraine IPS)
- Date range: March 2022 onwards
- View: Table
- Export: CSV

## Data Coverage

- **Start Date**: February 24, 2022 (synchronization date)
- **Reliable Data From**: March 1, 2022
- **Resolution**: Hourly (most data types)
- **Update Frequency**: Near real-time (1-2 hour delay)

## OSINT Applications

### Infrastructure Status Monitoring
- **Nuclear plants**: Check if Zaporizhzhia, Rivne, Khmelnytskyi, South Ukraine plants are generating
- **Thermal plants**: Coal and gas plant output indicates fuel availability
- **Hydro**: Kakhovka dam destruction visible in hydro generation data

### Blackout Detection
- Sudden load drops indicate power outages
- Compare actual load vs forecast to detect demand curtailment
- Regional data shows geographic impact of attacks

### Import Dependency Analysis
- High imports = domestic generation shortfall
- Cross-border flows spike during/after infrastructure attacks
- Emergency imports from Poland, Slovakia, Hungary, Romania

### Timeline Correlation
- Correlate generation drops with reported attacks
- Match load reductions with scheduled/emergency blackouts
- Track recovery times after infrastructure strikes

## Reference Files

- `api_request_template.json` - API request examples
- `ukraine_data_coverage.json` - Data coverage metadata
- `direct_urls.json` - Direct URLs for web access
- `DOWNLOAD_INSTRUCTIONS.txt` - Detailed download guide

## Data Sources

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- [ENTSO-E API Documentation](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html)
- [entsoe-py Python Package](https://github.com/EnergieID/entsoe-py)
- [Ukraine Energy Map](https://map.ua-energy.org/en/)

## License

ENTSO-E data is provided under the [Transparency Regulation (EU) No 543/2013](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32013R0543).

Data may be used for research, analysis, and non-commercial purposes with attribution.
