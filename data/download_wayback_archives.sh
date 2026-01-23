#!/bin/bash
# ============================================================
# Wayback Machine Archive Downloader
# ============================================================
# Downloads historical data from discontinued sources
# Run from: /Users/daniel.tipton/ML_OSINT/data/
# Usage: chmod +x download_wayback_archives.sh && ./download_wayback_archives.sh
# ============================================================

set -e
WAYBACK_DIR="/Users/daniel.tipton/ML_OSINT/data/wayback_archives"
LOG_FILE="$WAYBACK_DIR/wayback_download_log_$(date +%Y%m%d_%H%M%S).txt"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "$1" | tee -a "$LOG_FILE"; }
log_success() { log "${GREEN}[SUCCESS]${NC} $1"; }
log_info() { log "${YELLOW}[INFO]${NC} $1"; }
log_error() { log "${RED}[ERROR]${NC} $1"; }

# ============================================================
# Setup
# ============================================================
log_info "Creating directory structure..."
mkdir -p "$WAYBACK_DIR"/{google_mobility,apple_mobility,isw_assessments,cdx_indexes}
cd "$WAYBACK_DIR"

# ============================================================
# 1. Google Mobility Reports (Discontinued Oct 2022)
# ============================================================
log_info "=== Downloading Google Mobility Archives ==="
cd "$WAYBACK_DIR/google_mobility"

# Key dates during Ukraine conflict (Feb 24, 2022 invasion)
DATES=(
    "20220224"  # Invasion day
    "20220301"  # Week 1
    "20220315"  # 3 weeks
    "20220401"  # 5 weeks
    "20220501"  # 2+ months
    "20220601"  # 3+ months
    "20220701"  # 4+ months
    "20220801"  # 5+ months
    "20220901"  # 6+ months
    "20221001"  # 7+ months
    "20221015"  # Final (discontinued Oct 15)
)

for date in "${DATES[@]}"; do
    if [ -f "global_mobility_${date}.csv" ]; then
        log_info "Already exists: global_mobility_${date}.csv"
        continue
    fi

    log_info "Downloading Google Mobility: $date"
    if curl -L -f -o "global_mobility_${date}.csv" \
        "https://web.archive.org/web/${date}120000id_/https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv" 2>/dev/null; then
        log_success "Downloaded: global_mobility_${date}.csv ($(du -h global_mobility_${date}.csv | cut -f1))"
    else
        log_error "Failed: $date - trying alternative timestamp..."
        # Try midnight timestamp
        curl -L -f -o "global_mobility_${date}.csv" \
            "https://web.archive.org/web/${date}000000id_/https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv" 2>/dev/null || \
            log_error "  Still failed: $date"
    fi
    sleep 2  # Be nice to archive.org
done

# ============================================================
# 2. Apple Mobility Trends (Discontinued Apr 2022)
# ============================================================
log_info "=== Downloading Apple Mobility Archive ==="
cd "$WAYBACK_DIR/apple_mobility"

if [ -f "apple_mobility_final.csv" ]; then
    log_info "Already exists: apple_mobility_final.csv"
else
    log_info "Downloading Apple Mobility final snapshot..."
    if curl -L -f -o "apple_mobility_final.csv" \
        "https://web.archive.org/web/20220414120000id_/https://covid19-static.cdn-apple.com/covid19-mobility-data/current/applemobilitytrends.csv" 2>/dev/null; then
        log_success "Downloaded Apple Mobility ($(du -h apple_mobility_final.csv | cut -f1))"
    else
        log_error "Apple Mobility download failed"
        log_info "Alternative: git clone https://github.com/ActiveConclusion/COVID19_mobility.git"
    fi
fi

# Pre-invasion snapshot
if [ ! -f "apple_mobility_20220220.csv" ]; then
    log_info "Downloading Apple Mobility pre-invasion snapshot..."
    curl -L -f -o "apple_mobility_20220220.csv" \
        "https://web.archive.org/web/20220220120000id_/https://covid19-static.cdn-apple.com/covid19-mobility-data/current/applemobilitytrends.csv" 2>/dev/null || \
        log_error "Pre-invasion Apple Mobility failed"
fi

# ============================================================
# 3. CDX Indexes (for discovering more snapshots)
# ============================================================
log_info "=== Downloading CDX Indexes ==="
cd "$WAYBACK_DIR/cdx_indexes"

log_info "Getting Google Mobility CDX index..."
curl -sf "https://web.archive.org/cdx/search/cdx?url=gstatic.com/covid19/mobility/Global_Mobility*&output=json&limit=2000" \
    -o google_mobility_cdx.json && log_success "Google Mobility CDX saved"

log_info "Getting Apple Mobility CDX index..."
curl -sf "https://web.archive.org/cdx/search/cdx?url=covid19-static.cdn-apple.com/covid19-mobility-data/*&output=json&limit=500" \
    -o apple_mobility_cdx.json && log_success "Apple Mobility CDX saved"

log_info "Getting ISW assessments CDX index..."
curl -sf "https://web.archive.org/cdx/search/cdx?url=understandingwar.org/backgrounder/russian-offensive*&output=json&from=20220201&limit=2000" \
    -o isw_assessments_cdx.json && log_success "ISW CDX saved"

log_info "Getting NetBlocks Ukraine CDX index..."
curl -sf "https://web.archive.org/cdx/search/cdx?url=netblocks.org/*ukraine*&output=json&limit=1000" \
    -o netblocks_ukraine_cdx.json && log_success "NetBlocks CDX saved"

log_info "Getting Liveuamap CDX index..."
curl -sf "https://web.archive.org/cdx/search/cdx?url=liveuamap.com&output=json&from=20220201&limit=1000" \
    -o liveuamap_cdx.json && log_success "Liveuamap CDX saved"

# ============================================================
# 4. Extract Ukraine data from Google Mobility CSVs
# ============================================================
log_info "=== Extracting Ukraine-specific data ==="
cd "$WAYBACK_DIR/google_mobility"

mkdir -p ukraine_only
for csv in global_mobility_*.csv; do
    if [ -f "$csv" ]; then
        ukraine_file="ukraine_only/ukraine_${csv#global_mobility_}"
        if [ ! -f "$ukraine_file" ]; then
            log_info "Extracting Ukraine from $csv..."
            head -1 "$csv" > "$ukraine_file"
            grep "Ukraine" "$csv" >> "$ukraine_file" 2>/dev/null || true
            log_success "Created $ukraine_file"
        fi
    fi
done

# ============================================================
# Summary
# ============================================================
log_info "============================================================"
log_info "WAYBACK ARCHIVE DOWNLOAD SUMMARY"
log_info "============================================================"
echo ""
echo "Downloaded archives:"
echo ""
echo "Google Mobility (conflict period):"
ls -lh "$WAYBACK_DIR/google_mobility/"*.csv 2>/dev/null | head -15 || echo "  No files"
echo ""
echo "Apple Mobility:"
ls -lh "$WAYBACK_DIR/apple_mobility/"*.csv 2>/dev/null || echo "  No files"
echo ""
echo "CDX Indexes (for discovering more snapshots):"
ls -lh "$WAYBACK_DIR/cdx_indexes/"*.json 2>/dev/null || echo "  No files"
echo ""
echo "Ukraine-specific extracts:"
ls -lh "$WAYBACK_DIR/google_mobility/ukraine_only/"*.csv 2>/dev/null | head -15 || echo "  No files"
echo ""

# Calculate total size
TOTAL_SIZE=$(du -sh "$WAYBACK_DIR" | cut -f1)
log_info "Total archive size: $TOTAL_SIZE"
log_info "Log file: $LOG_FILE"
log_success "Wayback archive download complete!"

# ============================================================
# Usage notes
# ============================================================
echo ""
echo "============================================================"
echo "NEXT STEPS"
echo "============================================================"
echo ""
echo "1. Review CDX indexes to find additional snapshot dates:"
echo "   cat $WAYBACK_DIR/cdx_indexes/google_mobility_cdx.json | python3 -m json.tool"
echo ""
echo "2. Download additional snapshots using the timestamp from CDX:"
echo "   curl -L -o output.csv 'https://web.archive.org/web/{timestamp}id_/{url}'"
echo ""
echo "3. For ISW assessments, parse the CDX and download HTML pages"
echo ""
