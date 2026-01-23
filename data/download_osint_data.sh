#!/bin/bash
# ============================================================
# Ukraine Conflict OSINT Data Downloader
# ============================================================
# Downloads all freely available (no-account-required) data sources
# Run from: /Users/daniel.tipton/ML_OSINT/data/
# Usage: chmod +x download_osint_data.sh && ./download_osint_data.sh
# ============================================================

set -e
BASE_DIR="/Users/daniel.tipton/ML_OSINT/data"
LOG_FILE="$BASE_DIR/download_log_$(date +%Y%m%d_%H%M%S).txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_info() {
    log "${YELLOW}[INFO]${NC} $1"
}

# ============================================================
# Create directory structure
# ============================================================
log_info "Creating directory structure..."
mkdir -p "$BASE_DIR"/{viina,hdx,osm,reliefweb,maxar,ookla,alerts_ua,wayback_archives,acled_sample}

# ============================================================
# 1. VIINA 2.0 Dataset (Georgetown University)
# ============================================================
log_info "=== Downloading VIINA 2.0 Dataset ==="
cd "$BASE_DIR/viina"

if curl -L -f -o viina_main.zip "https://github.com/zhukovyuri/VIINA/archive/refs/heads/main.zip" 2>/dev/null; then
    unzip -o viina_main.zip && rm viina_main.zip
    log_success "VIINA 2.0 downloaded successfully"
else
    log_error "VIINA download failed - trying individual files..."

    # Try downloading individual data files
    for year in 2022 2023 2024 2025 2026; do
        curl -L -f -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/control_latest_${year}.zip" 2>/dev/null || true
        curl -L -f -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/event_1pd_latest_${year}.zip" 2>/dev/null || true
    done

    # GeoJSON files
    curl -L -f -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/gn_UA_tess.geojson" 2>/dev/null || true
    curl -L -f -O "https://github.com/zhukovyuri/VIINA/raw/main/Data/katotth_UA_tess.geojson" 2>/dev/null || true
fi

# ============================================================
# 2. HDX HAPI (Humanitarian API - no auth required)
# ============================================================
log_info "=== Downloading HDX HAPI Data ==="
cd "$BASE_DIR/hdx"

# Conflict events (ACLED integration)
if curl -sf "https://hapi.humdata.org/api/v2/coordination-context/conflict-events?location_code=UKR&output_format=csv&limit=100000" -o hdx_conflict_events.csv; then
    log_success "HDX conflict events downloaded"
else
    log_error "HDX conflict events failed"
fi

# Refugees
if curl -sf "https://hapi.humdata.org/api/v2/affected-people/refugees-persons-of-concern?origin_location_code=UKR&output_format=csv&limit=100000" -o hdx_refugees.csv; then
    log_success "HDX refugee data downloaded"
else
    log_error "HDX refugee data failed"
fi

# Humanitarian needs
if curl -sf "https://hapi.humdata.org/api/v2/affected-people/humanitarian-needs?location_code=UKR&output_format=csv&limit=100000" -o hdx_humanitarian_needs.csv; then
    log_success "HDX humanitarian needs downloaded"
else
    log_error "HDX humanitarian needs failed"
fi

# Operational presence (3W)
if curl -sf "https://hapi.humdata.org/api/v2/coordination-context/operational-presence?location_code=UKR&output_format=csv&limit=100000" -o hdx_operational_presence.csv; then
    log_success "HDX operational presence downloaded"
else
    log_error "HDX operational presence failed"
fi

# IDPs
if curl -sf "https://hapi.humdata.org/api/v2/affected-people/idps?location_code=UKR&output_format=csv&limit=100000" -o hdx_idps.csv; then
    log_success "HDX IDP data downloaded"
else
    log_error "HDX IDP data failed"
fi

# ============================================================
# 3. OpenStreetMap Ukraine Extract
# ============================================================
log_info "=== Downloading OpenStreetMap Ukraine ==="
cd "$BASE_DIR/osm"

if curl -L -f -o ukraine-latest.osm.pbf "https://download.geofabrik.de/europe/ukraine-latest.osm.pbf" 2>/dev/null; then
    log_success "OSM Ukraine extract downloaded (~1.5GB)"
else
    log_error "OSM download failed"
fi

# ============================================================
# 4. ReliefWeb API
# ============================================================
log_info "=== Downloading ReliefWeb Data ==="
cd "$BASE_DIR/reliefweb"

if curl -sf "https://api.reliefweb.int/v1/reports?appname=ml_osint&filter[field]=country.name&filter[value]=Ukraine&limit=1000&preset=latest" -o reliefweb_reports.json; then
    log_success "ReliefWeb reports downloaded"
else
    log_error "ReliefWeb reports failed"
fi

if curl -sf "https://api.reliefweb.int/v1/disasters?appname=ml_osint&filter[field]=country.name&filter[value]=Ukraine" -o reliefweb_disasters.json; then
    log_success "ReliefWeb disasters downloaded"
else
    log_error "ReliefWeb disasters failed"
fi

# ============================================================
# 5. Maxar/Vantor Open Data Catalog
# ============================================================
log_info "=== Downloading Maxar Open Data Catalog ==="
cd "$BASE_DIR/maxar"

if curl -sf -o maxar_catalog.json "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"; then
    log_success "Maxar STAC catalog downloaded"
else
    log_error "Maxar catalog failed - may have moved to Vantor"
fi

# ============================================================
# 6. Ukraine Air Raid Alerts - Public Data
# ============================================================
log_info "=== Downloading Ukraine Air Raid Alert Data ==="
cd "$BASE_DIR/alerts_ua"

# Region list (public)
if curl -sf "https://api.alerts.in.ua/v1/regions.json" -o regions.json; then
    log_success "Alert regions list downloaded"
else
    log_error "Alert regions failed"
fi

# Alternative: alerts.com.ua
if curl -sf "https://alerts.com.ua/api/states" -o alerts_current_states.json 2>/dev/null; then
    log_success "Current alert states downloaded (alerts.com.ua)"
else
    log_info "alerts.com.ua may require API key"
fi

# ============================================================
# 7. Ookla Open Data (via GitHub mirror)
# ============================================================
log_info "=== Downloading Ookla Open Data ==="
cd "$BASE_DIR/ookla"

if command -v git &> /dev/null; then
    if [ -d ".git" ]; then
        git pull 2>/dev/null && log_success "Ookla data updated"
    else
        git clone --depth 1 https://github.com/teamookla/ookla-open-data.git . 2>/dev/null && log_success "Ookla data cloned"
    fi
else
    log_info "Git not available - download manually from github.com/teamookla/ookla-open-data"
fi

# ============================================================
# 8. Wayback Machine - Historical Mobility Data
# ============================================================
log_info "=== Checking Wayback Machine for Historical Data ==="
cd "$BASE_DIR/wayback_archives"

# Check for Google Mobility snapshots
log_info "Google Mobility Reports (discontinued Oct 2022) - checking archives..."
curl -sf "https://web.archive.org/cdx/search/cdx?url=gstatic.com/covid19/mobility/*Ukraine*&output=json&limit=10" -o google_mobility_wayback_index.json 2>/dev/null || true

# Apple Mobility
log_info "Apple Mobility Trends (discontinued Apr 2022) - checking archives..."
curl -sf "https://web.archive.org/cdx/search/cdx?url=covid19.apple.com/mobility&output=json&limit=10" -o apple_mobility_wayback_index.json 2>/dev/null || true

# ============================================================
# Summary
# ============================================================
log_info "============================================================"
log_info "DOWNLOAD SUMMARY"
log_info "============================================================"

echo ""
log_info "Downloaded data locations:"
echo "  - VIINA 2.0:      $BASE_DIR/viina/"
echo "  - HDX HAPI:       $BASE_DIR/hdx/"
echo "  - OSM Ukraine:    $BASE_DIR/osm/"
echo "  - ReliefWeb:      $BASE_DIR/reliefweb/"
echo "  - Maxar Catalog:  $BASE_DIR/maxar/"
echo "  - Air Raid Alerts:$BASE_DIR/alerts_ua/"
echo "  - Ookla:          $BASE_DIR/ookla/"
echo "  - Wayback Index:  $BASE_DIR/wayback_archives/"
echo ""

log_info "============================================================"
log_info "SOURCES REQUIRING ACCOUNT REGISTRATION"
log_info "============================================================"
echo ""
echo "The following sources require free account registration:"
echo ""
echo "1. NASA Earthdata (Black Marble VNP46A2)"
echo "   Register: https://urs.earthdata.nasa.gov/users/new"
echo ""
echo "2. ACLED (Armed Conflict Data)"
echo "   Register: https://developer.acleddata.com/"
echo ""
echo "3. Copernicus Data Space (Sentinel-1/2)"
echo "   Register: https://dataspace.copernicus.eu/"
echo ""
echo "4. ERA5 Weather (Copernicus CDS)"
echo "   Register: https://cds.climate.copernicus.eu/user/register"
echo ""
echo "5. ENTSO-E Transparency (Power Grid)"
echo "   Register: https://transparency.entsoe.eu/"
echo ""
echo "6. Ukraine Air Raid API (for historical data)"
echo "   Request token: https://alerts.in.ua/api-request"
echo ""
echo "See OSINT_DATA_DOWNLOAD_GUIDE.md for detailed instructions."
echo ""

log_info "Log file: $LOG_FILE"
log_success "Download script completed!"
