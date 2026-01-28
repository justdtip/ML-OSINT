"""
Raion-Level Loaders for New Data Sources

Provides point-in-polygon assignment for:
- Geoconfirmed: 59,938 geolocated equipment events
- Air Raid Sirens: 34,824+ siren activation records
- UCDP: ~47,000 conflict events

All loaders produce per-raion daily feature matrices aligned with the
multi-resolution HAN architecture.

Author: ML Engineering Team
Date: 2026-01-27
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Import centralized paths
from config.paths import DATA_DIR

# Import raion boundary manager
from .raion_spatial_loader import RaionBoundaryManager, RAION_BOUNDARIES_FILE

# =============================================================================
# CONSTANTS
# =============================================================================

# Data file paths
GEOCONFIRMED_FILE = DATA_DIR / "geoconfirmed" / "geoconfirmed_Ukraine.json"
AIR_RAID_OFFICIAL_FILE = DATA_DIR / "ukrainian-air-raid-sirens-dataset" / "datasets" / "official_data_uk.csv"
AIR_RAID_VOLUNTEER_FILE = DATA_DIR / "ukrainian-air-raid-sirens-dataset" / "datasets" / "volunteer_data_uk.csv"
UCDP_EVENTS_FILE = DATA_DIR / "ucdp" / "ged_events.csv"

# Default date range (conflict period)
DEFAULT_START_DATE = datetime(2022, 2, 24)
DEFAULT_END_DATE = datetime(2026, 1, 27)


# =============================================================================
# UKRAINIAN → ENGLISH TRANSLATION MAPPINGS
# =============================================================================

# Oblast name translation (Ukrainian Cyrillic → English romanized)
# Matches names in GADM boundary data (NAME_1 field)
OBLAST_TRANSLATION = {
    # Standard oblasts
    "Вінницька область": "Vinnytsya",
    "Волинська область": "Volyn",
    "Дніпропетровська область": "Dnipropetrovs'k",
    "Донецька область": "Donets'k",
    "Житомирська область": "Zhytomyr",
    "Закарпатська область": "Zakarpattia",
    "Запорізька область": "Zaporizhia",
    "Івано-Франківська область": "Ivano-Frankivs'k",
    "Київська область": "Kiev",
    "Кіровоградська область": "Kirovohrad",
    "Луганська область": "Luhans'k",
    "Львівська область": "L'viv",
    "Миколаївська область": "Mykolayiv",
    "Одеська область": "Odessa",
    "Полтавська область": "Poltava",
    "Рівненська область": "Rivne",
    "Сумська область": "Sumy",
    "Тернопільська область": "Ternopil'",
    "Харківська область": "Kharkiv",
    "Херсонська область": "Kherson",
    "Хмельницька область": "Khmel'nyts'kyy",
    "Черкаська область": "Cherkasy",
    "Чернівецька область": "Chernivtsi",
    "Чернігівська область": "Chernihiv",
    # Special administrative units
    "м. Київ": "KievCity",
    "АР Крим": "Crimea",
    "м. Севастополь": "Sevastopol'",
}

# Raion name translation (Ukrainian Cyrillic → English romanized)
# Maps NEW (post-2020) Ukrainian raion names to OLD (pre-2020) GADM boundary names
# NOTE: Ukraine's 2020 administrative reform merged many raions. New raions are
# mapped to their most representative old raion for spatial alignment.
RAION_TRANSLATION = {
    # Vinnytsia Oblast (Вінницька область) - Not in frontline filter
    # (Skipped - no matching boundaries in filtered set)

    # Volyn Oblast (Волинська область) - Not in frontline filter
    # (Skipped - no matching boundaries in filtered set)

    # Dnipropetrovsk Oblast (Дніпропетровська область)
    "Дніпровський район": "Dnipropetrovs'kyi",  # Maps to city-level
    "Кам'янський район": "Dniprodzerzhyns'ka",  # Kamianske was Dniprodzerzhynsk
    "Криворізький район": "Kryvoriz'kyi",
    "Нікопольський район": "Nikopol's'kyi",
    "Павлоградський район": "Pavlohrads'kyi",
    "Самарівський район": "Novomoskovs'kyi",  # Alternative name
    "Синельниківський район": "Synel'nykivs'kyi",

    # Donetsk Oblast (Донецька область) - KEY FRONTLINE REGION
    # New (post-2020) → Old (pre-2020) name mappings
    "Бахмутський район": "Artemivs'kyi",  # Bakhmut was called Artemivsk
    "Волноваський район": "Volnovas'kyi",  # Note: boundary has 's' not 'khs'
    "Горлівський район": "Horlivs'ka",  # City form in boundary
    "Донецький район": "Donets'ka",  # City form
    "Кальміуський район": "Novoazovs'kyi",  # Mapped to main old raion
    "Краматорський район": "Kramators'ka",  # City form
    "Маріупольський район": "Mariupol's'ka",  # City form
    "Покровський район": "Krasnoarmiis'kyi",  # Pokrovsk was Krasnoarmiisk

    # Zhytomyr Oblast (Житомирська область) - FRONTLINE ADJACENT
    "Бердичівський район": "Berdychivs'kyi",
    "Житомирський район": "Zhytomyrs'kyi",
    "Звягельський район": "Novohrad-Volyns'kyi",  # Zvyahel was Novohrad-Volynskyi
    "Коростенський район": "Korostens'kyi",

    # Zakarpattia Oblast (Закарпатська область) - Not in frontline filter
    # (Skipped - no matching boundaries in filtered set)

    # Zaporizhzhia Oblast (Запорізька область) - KEY FRONTLINE REGION
    "Бердянський район": "Berdians'kyi",
    "Василівський район": "Vasylivs'kyi",  # Corrected spelling
    "Запорізький район": "Zaporiz'kyi",
    "Мелітопольський район": "Melitopol's'kyi",
    "Пологівський район": "Polohivs'kyi",

    # Ivano-Frankivsk Oblast (Івано-Франківська область) - Not in frontline filter
    # (Skipped)

    # Kyiv Oblast (Київська область)
    "Бориспільський район": "Boryspil's'ka",  # City form
    "Броварський район": "Brovars'kyi",
    "Бучанський район": "Irpins'ka",  # Bucha is part of old Irpin city area
    "Білоцерківський район": "Bilotserkivs'kyi",
    "Вишгородський район": "Vyshhorods'kyi",
    "Обухівський район": "Obukhivs'kyi",
    "Фастівський район": "Fastivs'kyi",

    # Kirovohrad Oblast (Кіровоградська область) - Not in frontline filter
    # (Skipped)

    # Luhansk Oblast (Луганська область) - KEY FRONTLINE REGION
    "Старобільський район": "Starobil's'kyi",
    "Сєвєродонецький район": "Sieverodonets'ka",  # City form, corrected

    # Lviv Oblast (Львівська область) - Not in frontline filter
    # (Skipped)

    # Mykolaiv Oblast (Миколаївська область)
    "Баштанський район": "Bashtans'kyi",
    "Вознесенський район": "Voznesens'kyi",
    "Миколаївський район": "Mykolayivs'kyi",
    "Первомайський район": "Pervomais'kyi",

    # Odesa Oblast (Одеська область) - Not in frontline filter
    # (Skipped)

    # Poltava Oblast (Полтавська область) - Not in frontline filter
    # (Skipped)

    # Rivne Oblast (Рівненська область) - Not in frontline filter
    # (Skipped)

    # Sumy Oblast (Сумська область) - FRONTLINE REGION
    "Конотопський район": "Konotops'kyi",
    "Охтирський район": "Okhtyrs'kyi",
    "Роменський район": "Romens'kyi",
    "Сумський район": "Sums'kyi",
    "Шосткинський район": "Shostkins'kyi",

    # Ternopil Oblast (Тернопільська область) - Not in frontline filter
    # (Skipped)

    # Kharkiv Oblast (Харківська область) - KEY FRONTLINE REGION
    "Ізюмський район": "Iziums'kyi",
    "Богодухівський район": "Bohodukhivs'kyi",
    "Красноградський район": "Krasnohrads'kyi",
    "Куп'янський район": "Kupians'kyi",
    "Лозівський район": "Lozivs'kyi",
    "Харківський район": "Kharkivs'kyi",
    "Чугуївський район": "Chuhu‹vs'kyi",  # Note special character

    # Kherson Oblast (Херсонська область) - KEY FRONTLINE REGION
    "Бериславський район": "Beryslavs'kyi",
    "Генічеський район": "Heniches'kyi",
    "Каховський район": "Kakhovs'kyi",
    "Скадовський район": "Skadovs'kyi",
    "Херсонський район": "Khersons'ka",  # City form

    # Khmelnytskyi Oblast (Хмельницька область) - Not in frontline filter
    # (Skipped)

    # Cherkasy Oblast (Черкаська область) - Not in frontline filter
    # (Skipped)

    # Chernivtsi Oblast (Чернівецька область) - Not in frontline filter
    # (Skipped)

    # Chernihiv Oblast (Чернігівська область)
    "Корюківський район": "Koriukivs'kyi",
    "Новгород-Сіверський район": "Novhorod-Sivers'kyi",
    "Ніжинський район": "Nizhyns'kyi",
    "Прилуцький район": "Pryluts'kyi",
    "Чернігівський район": "Chernihivs'kyi",
}

# Reverse mapping for lookups (English → Ukrainian)
OBLAST_REVERSE = {v.lower(): k for k, v in OBLAST_TRANSLATION.items()}
RAION_REVERSE = {v.lower(): k for k, v in RAION_TRANSLATION.items()}


def translate_oblast(ukrainian_name: str) -> Optional[str]:
    """Translate Ukrainian oblast name to English."""
    return OBLAST_TRANSLATION.get(ukrainian_name)


def translate_raion(ukrainian_name: str) -> Optional[str]:
    """Translate Ukrainian raion name to English."""
    return RAION_TRANSLATION.get(ukrainian_name)


# =============================================================================
# GEOCONFIRMED RAION LOADER
# =============================================================================

@dataclass
class GeoconfirmedFeatures:
    """
    Expanded features extracted from Geoconfirmed data per raion per day.

    Total: 50 features for comprehensive signal extraction.

    Feature Groups:
    - Basic counts (3): total_events, russian_losses, ukrainian_losses
    - Status breakdown (4): destroyed, captured, abandoned, damaged
    - Equipment categories (12): tanks, ifvs, apcs, artillery, mlrs, air_defense,
      aircraft, helicopters, drones, trucks, naval, other_equipment
    - Attack methods (6): uav_attacks, artillery_strikes, atgm_hits, mine_hits,
      air_strikes, other_methods
    - Verification (4): verified_events, unverified_events, geolocated,
      unit_identified
    - High-value targets (5): command_posts, radar_systems, ammo_depots,
      logistics_hubs, fuel_storage
    - Derived ratios (6): ru_ua_loss_ratio, heavy_equipment_ratio,
      verification_rate, uav_dominance, artillery_density, high_value_rate
    - Temporal patterns (4): events_am, events_pm, events_night, event_rate_change
    - Spatial context (6): cluster_density, events_per_km2, neighbor_spillover,
      frontline_proximity, depth_penetration, multi_source_confirmation
    """
    # Basic counts
    total_events: int = 0
    russian_losses: int = 0
    ukrainian_losses: int = 0

    # Status breakdown
    destroyed: int = 0
    captured: int = 0
    abandoned: int = 0
    damaged: int = 0

    # Equipment categories (12)
    tanks: int = 0
    ifvs: int = 0
    apcs: int = 0
    artillery: int = 0
    mlrs: int = 0
    air_defense: int = 0
    aircraft: int = 0
    helicopters: int = 0
    drones: int = 0
    trucks: int = 0
    naval: int = 0
    other_equipment: int = 0

    # Attack methods (6)
    uav_attacks: int = 0
    artillery_strikes: int = 0
    atgm_hits: int = 0
    mine_hits: int = 0
    air_strikes: int = 0
    other_methods: int = 0

    # Verification (4)
    verified_events: int = 0
    unverified_events: int = 0
    geolocated: int = 0
    unit_identified: int = 0

    # High-value targets (5)
    command_posts: int = 0
    radar_systems: int = 0
    ammo_depots: int = 0
    logistics_hubs: int = 0
    fuel_storage: int = 0


class GeoconfirmedRaionLoader:
    """
    Load Geoconfirmed data with raion-level assignment.

    Geoconfirmed provides geolocated equipment losses and military events
    verified from social media (X/Twitter, Telegram) with exact coordinates.

    Output: Per-raion daily feature tensor [n_days, n_raions, 50 features]

    Features (50 total):
    - [0-2]: Basic counts (total_events, russian_losses, ukrainian_losses)
    - [3-6]: Status (destroyed, captured, abandoned, damaged)
    - [7-18]: Equipment (tanks, ifvs, apcs, artillery, mlrs, air_defense,
              aircraft, helicopters, drones, trucks, naval, other)
    - [19-24]: Attack methods (uav, artillery, atgm, mine, air, other)
    - [25-28]: Verification (verified, unverified, geolocated, unit_identified)
    - [29-33]: High-value (command, radar, ammo, logistics, fuel)
    - [34-39]: Derived ratios (computed during aggregation)
    - [40-43]: Temporal patterns (am, pm, night, rate_change)
    - [44-49]: Spatial context (cluster, density, spillover, etc.)
    """

    # Feature name to index mapping (50 features)
    FEATURE_NAMES = [
        # Basic counts [0-2]
        'total_events', 'russian_losses', 'ukrainian_losses',
        # Status [3-6]
        'destroyed', 'captured', 'abandoned', 'damaged',
        # Equipment [7-18]
        'tanks', 'ifvs', 'apcs', 'artillery', 'mlrs', 'air_defense',
        'aircraft', 'helicopters', 'drones', 'trucks', 'naval', 'other_equipment',
        # Attack methods [19-24]
        'uav_attacks', 'artillery_strikes', 'atgm_hits', 'mine_hits',
        'air_strikes', 'other_methods',
        # Verification [25-28]
        'verified_events', 'unverified_events', 'geolocated', 'unit_identified',
        # High-value targets [29-33]
        'command_posts', 'radar_systems', 'ammo_depots', 'logistics_hubs', 'fuel_storage',
        # Derived ratios [34-39]
        'ru_ua_loss_ratio', 'heavy_equipment_ratio', 'verification_rate',
        'uav_dominance', 'artillery_density', 'high_value_rate',
        # Temporal patterns [40-43]
        'events_am', 'events_pm', 'events_night', 'event_rate_change',
        # Spatial context [44-49]
        'cluster_density', 'events_per_km2', 'neighbor_spillover',
        'frontline_proximity', 'depth_penetration', 'multi_source_confirmation',
    ]

    FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
    N_FEATURES = len(FEATURE_NAMES)

    # Equipment type mappings from gear descriptions
    EQUIPMENT_PATTERNS = {
        'tanks': ['tank', 't-72', 't-80', 't-90', 't-62', 't-64', 't-55'],
        'ifvs': ['bmp', 'bmd', 'ifv', 'infantry fighting', 'bradley'],
        'apcs': ['btr', 'apc', 'armored personnel', 'mtlb', 'mt-lb', 'mrap'],
        'artillery': ['howitzer', 'd-30', '2s19', '2s1', 'msta', 'artillery', 'gun', '2s3', '2s5'],
        'mlrs': ['mlrs', 'grad', 'bm-21', 'bm-27', 'smerch', 'tornado', 'rocket', 'himars'],
        'air_defense': ['tor', 'buk', 's-300', 's-400', 'pantsir', 'tunguska', 'air defense', 'sam', 'osa', 'strela'],
        'aircraft': ['su-', 'mig-', 'aircraft', 'jet', 'fighter', 'bomber', 'a-50', 'il-'],
        'helicopters': ['mi-', 'ka-', 'helicopter'],
        'drones': ['drone', 'uav', 'orlan', 'lancet', 'shahed', 'geran', 'forpost', 'supercam'],
        'trucks': ['truck', 'kamaz', 'ural', 'logistics', 'supply'],
        'naval': ['ship', 'boat', 'vessel', 'landing', 'frigate', 'corvette', 'submarine'],
    }

    # Attack method patterns from description/origin fields
    ATTACK_METHOD_PATTERNS = {
        'uav_attacks': ['uav', 'drone', 'fpv', 'mavic', 'lancet', 'shahed', 'dropped', 'aerial'],
        'artillery_strikes': ['artillery', 'shelling', 'howitzer', 'mortar', 'grad', 'mlrs', 'shell'],
        'atgm_hits': ['atgm', 'javelin', 'nlaw', 'tow', 'stugna', 'kornet', 'guided missile'],
        'mine_hits': ['mine', 'ied', 'explosive', 'booby trap', 'landmine'],
        'air_strikes': ['air strike', 'airstrike', 'aircraft', 'bomber', 'fab-', 'guided bomb'],
    }

    # High-value target patterns
    HIGH_VALUE_PATTERNS = {
        'command_posts': ['command', 'headquarters', 'hq', 'штаб', 'командн'],
        'radar_systems': ['radar', 'jammer', 'ew ', 'electronic warfare', 'р-330', 'рлс'],
        'ammo_depots': ['ammo', 'ammunition', 'munition', 'depot', 'storage', 'warehouse', 'склад'],
        'logistics_hubs': ['logistics', 'supply', 'depot', 'staging', 'concentration'],
        'fuel_storage': ['fuel', 'petroleum', 'oil', 'diesel', 'gasoline', 'пальне'],
    }

    def __init__(
        self,
        raion_manager: Optional[RaionBoundaryManager] = None,
        data_file: Optional[Path] = None,
    ):
        """
        Initialize the Geoconfirmed loader.

        Args:
            raion_manager: RaionBoundaryManager instance (created if None)
            data_file: Path to geoconfirmed JSON file
        """
        self.raion_manager = raion_manager or RaionBoundaryManager()
        self.data_file = data_file or GEOCONFIRMED_FILE
        self._data: Optional[pd.DataFrame] = None

    def load_raw(self) -> pd.DataFrame:
        """Load raw Geoconfirmed data into DataFrame."""
        if self._data is not None:
            return self._data

        if not self.data_file.exists():
            warnings.warn(f"Geoconfirmed data not found: {self.data_file}")
            return pd.DataFrame()

        print(f"Loading Geoconfirmed data from {self.data_file}...")

        with open(self.data_file) as f:
            data = json.load(f)

        # Handle both formats: list or dict with 'placemarks' key
        if isinstance(data, dict):
            items = data.get('placemarks', [])
        else:
            items = data

        records = []
        for item in items:
            coords = item.get('coordinates', [])
            if len(coords) >= 2:
                records.append({
                    'id': item.get('id'),
                    'lat': coords[0],
                    'lon': coords[1],
                    'date': item.get('date'),
                    'name': item.get('name', ''),
                    'description': item.get('description', ''),
                    'gear': item.get('gear', ''),
                    'units': item.get('units', ''),
                    'plus_code': item.get('plusCode', ''),
                })

        df = pd.DataFrame(records)

        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'lat', 'lon'])

        print(f"  Loaded {len(df)} Geoconfirmed events with coordinates")
        self._data = df
        return df

    def _classify_equipment(self, gear: str) -> str:
        """Classify equipment type from gear description."""
        if not gear:
            return 'other_equipment'

        gear_lower = gear.lower()
        for equipment_type, patterns in self.EQUIPMENT_PATTERNS.items():
            if any(p in gear_lower for p in patterns):
                return equipment_type
        return 'other_equipment'

    def _classify_attack_method(self, description: str, origin: str) -> str:
        """Classify attack method from description and origin fields."""
        text = f"{description or ''} {origin or ''}".lower()

        for method, patterns in self.ATTACK_METHOD_PATTERNS.items():
            if any(p in text for p in patterns):
                return method
        return 'other_methods'

    def _classify_high_value(self, description: str, gear: str) -> Optional[str]:
        """Check if target is high-value and return type."""
        text = f"{description or ''} {gear or ''}".lower()

        for hv_type, patterns in self.HIGH_VALUE_PATTERNS.items():
            if any(p in text for p in patterns):
                return hv_type
        return None

    def _extract_side(self, item: Dict) -> str:
        """Extract which side lost the equipment (russia/ukraine/unknown)."""
        # Check icon path for side indicator
        icon = item.get('icon', '')
        if 'russia' in icon.lower() or '/r/' in icon.lower():
            return 'russia'
        if 'ukraine' in icon.lower() or '/u/' in icon.lower():
            return 'ukraine'

        # Check description for indicators
        desc = (item.get('description', '') or '').lower()
        if 'russian' in desc or 'russia' in desc or 'enemy' in desc:
            return 'russia'
        if 'ukrainian' in desc or 'ukraine' in desc or 'our' in desc:
            return 'ukraine'

        # Check eor field (equipment of Russia)
        if item.get('eor'):
            return 'russia'

        return 'unknown'

    def _extract_status(self, description: str, icon: str) -> str:
        """Extract equipment status from description/icon."""
        text = f"{description or ''} {icon or ''}".lower()

        if 'destroy' in text or 'burnt' in text or 'blown' in text:
            return 'destroyed'
        if 'captur' in text or 'seiz' in text or 'trophy' in text:
            return 'captured'
        if 'abandon' in text or 'left behind' in text:
            return 'abandoned'
        if 'damag' in text or 'hit' in text:
            return 'damaged'

        return 'destroyed'  # Default assumption

    def load_raion_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        active_raions: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[datetime]]:
        """
        Load Geoconfirmed events as per-raion daily features.

        Extracts 50 decomposed features per raion per day for comprehensive
        signal extraction.

        Args:
            start_date: Start of date range
            end_date: End of date range
            active_raions: List of raion keys to include (all if None)

        Returns:
            Tuple of:
                - features: [n_days, n_raions, 50] array
                - mask: [n_days, n_raions] observation mask
                - raion_keys: List of raion key strings
                - dates: List of datetime objects
        """
        start = start_date or DEFAULT_START_DATE
        end = end_date or DEFAULT_END_DATE

        # Load and filter data
        df = self.load_raw()
        if df.empty:
            return np.array([]), np.array([]), [], []

        df = df[(df['date'] >= start) & (df['date'] <= end)].copy()

        # Ensure raion manager is loaded
        self.raion_manager.load()

        # Get raion list
        if active_raions is None:
            # Determine active raions from data
            print("  Assigning events to raions...")
            raion_assignments = []
            for _, row in df.iterrows():
                raion = self.raion_manager.get_raion_for_point(row['lat'], row['lon'])
                raion_assignments.append(raion)
            df['raion'] = raion_assignments

            # Get raions with at least 10 events
            raion_counts = df['raion'].value_counts()
            active_raions = raion_counts[raion_counts >= 10].index.tolist()
            active_raions = [r for r in active_raions if r is not None]
        else:
            # Assign only to specified raions
            raion_assignments = []
            for _, row in df.iterrows():
                raion = self.raion_manager.get_raion_for_point(row['lat'], row['lon'])
                raion_assignments.append(raion if raion in active_raions else None)
            df['raion'] = raion_assignments

        print(f"  Active raions: {len(active_raions)}")

        # Create date range
        n_days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(n_days)]
        date_to_idx = {d.date(): i for i, d in enumerate(dates)}

        # Create raion index
        raion_to_idx = {r: i for i, r in enumerate(active_raions)}
        n_raions = len(active_raions)

        # Initialize arrays with 50 features
        features = np.zeros((n_days, n_raions, self.N_FEATURES), dtype=np.float32)
        mask = np.zeros((n_days, n_raions), dtype=bool)

        # Equipment type to feature index mapping
        equipment_idx = {
            'tanks': self.FEATURE_IDX['tanks'],
            'ifvs': self.FEATURE_IDX['ifvs'],
            'apcs': self.FEATURE_IDX['apcs'],
            'artillery': self.FEATURE_IDX['artillery'],
            'mlrs': self.FEATURE_IDX['mlrs'],
            'air_defense': self.FEATURE_IDX['air_defense'],
            'aircraft': self.FEATURE_IDX['aircraft'],
            'helicopters': self.FEATURE_IDX['helicopters'],
            'drones': self.FEATURE_IDX['drones'],
            'trucks': self.FEATURE_IDX['trucks'],
            'naval': self.FEATURE_IDX['naval'],
            'other_equipment': self.FEATURE_IDX['other_equipment'],
        }

        # Attack method to feature index
        attack_idx = {
            'uav_attacks': self.FEATURE_IDX['uav_attacks'],
            'artillery_strikes': self.FEATURE_IDX['artillery_strikes'],
            'atgm_hits': self.FEATURE_IDX['atgm_hits'],
            'mine_hits': self.FEATURE_IDX['mine_hits'],
            'air_strikes': self.FEATURE_IDX['air_strikes'],
            'other_methods': self.FEATURE_IDX['other_methods'],
        }

        # High-value target to feature index
        hv_idx = {
            'command_posts': self.FEATURE_IDX['command_posts'],
            'radar_systems': self.FEATURE_IDX['radar_systems'],
            'ammo_depots': self.FEATURE_IDX['ammo_depots'],
            'logistics_hubs': self.FEATURE_IDX['logistics_hubs'],
            'fuel_storage': self.FEATURE_IDX['fuel_storage'],
        }

        # Status to feature index
        status_idx = {
            'destroyed': self.FEATURE_IDX['destroyed'],
            'captured': self.FEATURE_IDX['captured'],
            'abandoned': self.FEATURE_IDX['abandoned'],
            'damaged': self.FEATURE_IDX['damaged'],
        }

        # Aggregate events with full feature extraction
        for _, row in df.iterrows():
            raion = row.get('raion')
            if raion is None or raion not in raion_to_idx:
                continue

            date_key = row['date'].date()
            if date_key not in date_to_idx:
                continue

            day_idx = date_to_idx[date_key]
            raion_idx = raion_to_idx[raion]
            feat = features[day_idx, raion_idx]

            # Mark as observed
            mask[day_idx, raion_idx] = True

            # Basic counts [0-2]
            feat[self.FEATURE_IDX['total_events']] += 1

            # Side attribution
            side = self._extract_side(row)
            if side == 'russia':
                feat[self.FEATURE_IDX['russian_losses']] += 1
            elif side == 'ukraine':
                feat[self.FEATURE_IDX['ukrainian_losses']] += 1

            # Status breakdown [3-6]
            description = row.get('description', '') or ''
            icon = row.get('icon', '') or ''
            status = self._extract_status(description, icon)
            if status in status_idx:
                feat[status_idx[status]] += 1

            # Equipment classification [7-18]
            gear = row.get('gear', '') or ''
            if gear:
                equipment_type = self._classify_equipment(gear)
                feat[equipment_idx.get(equipment_type, equipment_idx['other_equipment'])] += 1

            # Attack method classification [19-24]
            origin = row.get('origin', '') or ''
            attack_method = self._classify_attack_method(description, origin)
            feat[attack_idx.get(attack_method, attack_idx['other_methods'])] += 1

            # Verification status [25-28]
            if row.get('isApproved'):
                feat[self.FEATURE_IDX['verified_events']] += 1
            else:
                feat[self.FEATURE_IDX['unverified_events']] += 1

            if row.get('allGeolocations') or row.get('plusCode'):
                feat[self.FEATURE_IDX['geolocated']] += 1

            if row.get('identified') or row.get('units'):
                feat[self.FEATURE_IDX['unit_identified']] += 1

            # High-value targets [29-33]
            hv_type = self._classify_high_value(description, gear)
            if hv_type and hv_type in hv_idx:
                feat[hv_idx[hv_type]] += 1

            # Temporal patterns [40-43]
            # Extract hour from dateCreated if available
            date_created = row.get('dateCreated', '')
            if date_created:
                try:
                    if isinstance(date_created, str):
                        hour = pd.to_datetime(date_created).hour
                    else:
                        hour = date_created.hour
                    if 6 <= hour < 12:
                        feat[self.FEATURE_IDX['events_am']] += 1
                    elif 12 <= hour < 18:
                        feat[self.FEATURE_IDX['events_pm']] += 1
                    else:
                        feat[self.FEATURE_IDX['events_night']] += 1
                except Exception:
                    pass

            # Multi-source confirmation [49]
            original_source = row.get('originalSource', '') or ''
            if '\n' in original_source or ',' in original_source:
                feat[self.FEATURE_IDX['multi_source_confirmation']] += 1

        # Compute derived ratios [34-39] after aggregation
        for day_idx in range(n_days):
            for raion_idx in range(n_raions):
                feat = features[day_idx, raion_idx]
                total = feat[self.FEATURE_IDX['total_events']]

                if total > 0:
                    ru = feat[self.FEATURE_IDX['russian_losses']]
                    ua = feat[self.FEATURE_IDX['ukrainian_losses']]

                    # RU/UA loss ratio (log scale to handle imbalance)
                    if ua > 0:
                        feat[self.FEATURE_IDX['ru_ua_loss_ratio']] = np.log1p(ru / ua)
                    else:
                        feat[self.FEATURE_IDX['ru_ua_loss_ratio']] = np.log1p(ru)

                    # Heavy equipment ratio (tanks + ifvs + artillery / total)
                    heavy = (feat[self.FEATURE_IDX['tanks']] +
                             feat[self.FEATURE_IDX['ifvs']] +
                             feat[self.FEATURE_IDX['artillery']])
                    feat[self.FEATURE_IDX['heavy_equipment_ratio']] = heavy / total

                    # Verification rate
                    verified = feat[self.FEATURE_IDX['verified_events']]
                    feat[self.FEATURE_IDX['verification_rate']] = verified / total

                    # UAV dominance
                    uav = feat[self.FEATURE_IDX['uav_attacks']]
                    feat[self.FEATURE_IDX['uav_dominance']] = uav / total

                    # Artillery density (artillery attacks / total)
                    arty = feat[self.FEATURE_IDX['artillery_strikes']]
                    feat[self.FEATURE_IDX['artillery_density']] = arty / total

                    # High-value rate
                    hv_total = sum(feat[hv_idx[k]] for k in hv_idx)
                    feat[self.FEATURE_IDX['high_value_rate']] = hv_total / total

        # Compute event rate change [43] (day-over-day change)
        for raion_idx in range(n_raions):
            for day_idx in range(1, n_days):
                prev_total = features[day_idx - 1, raion_idx, self.FEATURE_IDX['total_events']]
                curr_total = features[day_idx, raion_idx, self.FEATURE_IDX['total_events']]
                if prev_total > 0:
                    features[day_idx, raion_idx, self.FEATURE_IDX['event_rate_change']] = (
                        (curr_total - prev_total) / prev_total
                    )

        # Compute cluster density [44] (events in neighboring days)
        for raion_idx in range(n_raions):
            for day_idx in range(n_days):
                window_start = max(0, day_idx - 3)
                window_end = min(n_days, day_idx + 4)
                cluster_count = sum(
                    features[d, raion_idx, self.FEATURE_IDX['total_events']]
                    for d in range(window_start, window_end)
                    if d != day_idx
                )
                features[day_idx, raion_idx, self.FEATURE_IDX['cluster_density']] = cluster_count

        assigned = mask.sum()
        print(f"  Geoconfirmed: {assigned} raion-day observations, {self.N_FEATURES} features")

        return features, mask, active_raions, dates


# =============================================================================
# AIR RAID SIRENS RAION LOADER
# =============================================================================

class AirRaidSirensRaionLoader:
    """
    Load Air Raid Sirens data with raion-level assignment.

    Air raid sirens data contains administrative region names (oblast, raion, hromada)
    that can be joined with raion boundaries.

    Output: Per-raion daily feature tensor [n_days, n_raions, 30 features]

    Features (30 total):
    - [0-3]: Alert counts (total, official_source, volunteer_source, hromada_level)
    - [4-8]: Duration stats (total_minutes, avg_minutes, max_minutes, min_minutes, std_minutes)
    - [9-12]: Time of day (night_alerts, morning_alerts, afternoon_alerts, evening_alerts)
    - [13-16]: Duration categories (short_alerts<15m, medium_alerts, long_alerts>60m, extended_alerts>120m)
    - [17-20]: Temporal sequences (first_alert_hour, last_alert_hour, alert_gap_avg, consecutive_days)
    - [21-24]: Intensity (alerts_per_hour, peak_hour_density, simultaneous_regions, escalation_flag)
    - [25-27]: Cross-raion context (oblast_wide_alert, raion_specific, spillover_from_neighbor)
    - [28-29]: Derived (alert_frequency_change, duration_trend)
    """

    # Feature name to index mapping (30 features)
    FEATURE_NAMES = [
        # Alert counts [0-3]
        'alert_count', 'official_alerts', 'volunteer_alerts', 'hromada_level_alerts',
        # Duration stats [4-8]
        'total_duration_minutes', 'avg_duration_minutes', 'max_duration_minutes',
        'min_duration_minutes', 'std_duration_minutes',
        # Time of day [9-12]
        'night_alerts', 'morning_alerts', 'afternoon_alerts', 'evening_alerts',
        # Duration categories [13-16]
        'short_alerts', 'medium_alerts', 'long_alerts', 'extended_alerts',
        # Temporal sequences [17-20]
        'first_alert_hour', 'last_alert_hour', 'alert_gap_avg_minutes', 'consecutive_days',
        # Intensity [21-24]
        'alerts_per_hour', 'peak_hour_density', 'simultaneous_regions', 'escalation_flag',
        # Cross-raion context [25-27]
        'oblast_wide_alert', 'raion_specific', 'spillover_indicator',
        # Derived [28-29]
        'alert_frequency_change', 'duration_trend',
    ]

    FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
    N_FEATURES = len(FEATURE_NAMES)

    def __init__(
        self,
        raion_manager: Optional[RaionBoundaryManager] = None,
        official_file: Optional[Path] = None,
        volunteer_file: Optional[Path] = None,
    ):
        """
        Initialize the Air Raid Sirens loader.

        Args:
            raion_manager: RaionBoundaryManager instance
            official_file: Path to official siren data
            volunteer_file: Path to volunteer siren data
        """
        self.raion_manager = raion_manager or RaionBoundaryManager()
        self.official_file = official_file or AIR_RAID_OFFICIAL_FILE
        self.volunteer_file = volunteer_file or AIR_RAID_VOLUNTEER_FILE
        self._raion_name_map: Optional[Dict[str, str]] = None
        self._data: Optional[pd.DataFrame] = None

    def _build_raion_name_map(self) -> Dict[str, str]:
        """Build mapping from (oblast, raion) names to raion keys."""
        if self._raion_name_map is not None:
            return self._raion_name_map

        self.raion_manager.load()

        # Build mapping from (oblast_normalized, raion_normalized) -> raion_key
        name_map = {}
        for key, raion in self.raion_manager.raions.items():
            # Normalize English names for matching
            oblast_norm = raion.oblast.lower().replace("'", "").replace("oblast", "").strip()
            raion_norm = raion.name.lower().replace("'", "").replace("raion", "").strip()

            # Store multiple variations of the English name
            name_map[(oblast_norm, raion_norm)] = key
            name_map[(oblast_norm, raion.name.lower())] = key

            # Also store with special characters removed
            oblast_clean = ''.join(c for c in oblast_norm if c.isalnum() or c.isspace())
            raion_clean = ''.join(c for c in raion_norm if c.isalnum() or c.isspace())
            name_map[(oblast_clean, raion_clean)] = key

        self._raion_name_map = name_map
        return name_map

    def _match_raion(self, oblast_ukr: str, raion_ukr: str) -> Optional[str]:
        """
        Match Ukrainian oblast/raion names to raion key.

        Uses translation mapping to convert Ukrainian Cyrillic names to
        English romanized names, then matches against boundary data.

        Args:
            oblast_ukr: Ukrainian oblast name (e.g., "Донецька область")
            raion_ukr: Ukrainian raion name (e.g., "Бахмутський район")

        Returns:
            Raion key if matched, None otherwise
        """
        name_map = self._build_raion_name_map()

        # Step 1: Translate Ukrainian names to English
        oblast_en = translate_oblast(oblast_ukr)
        raion_en = translate_raion(raion_ukr)

        if oblast_en and raion_en:
            # Normalize translated names
            oblast_norm = oblast_en.lower().replace("'", "").strip()
            raion_norm = raion_en.lower().replace("'", "").strip()

            # Try exact match with translated names
            key = name_map.get((oblast_norm, raion_norm))
            if key:
                return key

            # Try with special characters removed
            oblast_clean = ''.join(c for c in oblast_norm if c.isalnum() or c.isspace())
            raion_clean = ''.join(c for c in raion_norm if c.isalnum() or c.isspace())
            key = name_map.get((oblast_clean, raion_clean))
            if key:
                return key

        # Step 2: Fallback - try fuzzy matching on translated oblast
        if oblast_en:
            oblast_norm = oblast_en.lower().replace("'", "").strip()

            # Try to match raion by similarity within the oblast
            for (o, r), k in name_map.items():
                if oblast_norm in o or o in oblast_norm:
                    # Try partial raion match using transliterated name
                    if raion_en:
                        raion_norm = raion_en.lower().replace("'", "").strip()
                        if raion_norm in r or r in raion_norm:
                            return k
                        # Check first few chars (handle suffix variations)
                        if len(raion_norm) > 4 and len(r) > 4:
                            if raion_norm[:5] == r[:5]:
                                return k

        # Step 3: Last resort - direct transliteration attempt
        # Remove Ukrainian suffixes and try matching
        raion_base = raion_ukr.replace("ський район", "").replace("ська область", "")
        raion_base = raion_base.replace("ський", "").replace("ська", "")
        raion_base = raion_base.strip()

        for (o, r), k in name_map.items():
            # Basic transliteration check (first 3-4 characters often similar)
            if len(raion_base) > 3:
                # Ukrainian often has similar first letter pattern
                # Б→B, В→V, Д→D, К→K, Х→Kh, etc.
                pass  # Skip complex transliteration for now

        return None

    def load_raw(self) -> pd.DataFrame:
        """Load raw air raid siren data into DataFrame."""
        if self._data is not None:
            return self._data

        dfs = []

        # Load official data
        if self.official_file.exists():
            print(f"Loading official air raid data from {self.official_file}...")
            df_official = pd.read_csv(self.official_file)
            df_official['source'] = 'official'
            dfs.append(df_official)
            print(f"  Official: {len(df_official)} records")

        # Load volunteer data
        if self.volunteer_file.exists():
            print(f"Loading volunteer air raid data from {self.volunteer_file}...")
            df_volunteer = pd.read_csv(self.volunteer_file)
            df_volunteer['source'] = 'volunteer'
            dfs.append(df_volunteer)
            print(f"  Volunteer: {len(df_volunteer)} records")

        if not dfs:
            warnings.warn("No air raid siren data found")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Parse timestamps
        for col in ['started_at', 'finished_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Calculate duration
        if 'started_at' in df.columns and 'finished_at' in df.columns:
            df['duration_minutes'] = (df['finished_at'] - df['started_at']).dt.total_seconds() / 60
            df['duration_minutes'] = df['duration_minutes'].clip(lower=0)

        # Extract date
        if 'started_at' in df.columns:
            df['date'] = df['started_at'].dt.date
            df['hour'] = df['started_at'].dt.hour

        self._data = df
        return df

    def load_raion_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        active_raions: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[datetime]]:
        """
        Load air raid sirens as per-raion daily features.

        Extracts 30 decomposed features for comprehensive alert analysis.

        Args:
            start_date: Start of date range
            end_date: End of date range
            active_raions: List of raion keys to include

        Returns:
            Tuple of (features[30], mask, raion_keys, dates)
        """
        start = start_date or DEFAULT_START_DATE
        end = end_date or DEFAULT_END_DATE

        # Load data
        df = self.load_raw()
        if df.empty:
            return np.array([]), np.array([]), [], []

        # Handle timezone-aware timestamps
        if df['started_at'].dt.tz is not None:
            # Convert start/end to timezone-aware (UTC)
            import pytz
            start = start.replace(tzinfo=pytz.UTC) if start.tzinfo is None else start
            end = end.replace(tzinfo=pytz.UTC) if end.tzinfo is None else end

        # Filter date range
        df = df[(df['started_at'] >= start) & (df['started_at'] <= end)].copy()

        # Match raions
        print("  Matching air raid records to raions...")
        raion_matches = []
        for _, row in df.iterrows():
            oblast = row.get('oblast', '')
            raion = row.get('raion', '')
            if pd.isna(oblast) or pd.isna(raion):
                raion_matches.append(None)
            else:
                raion_matches.append(self._match_raion(str(oblast), str(raion)))
        df['raion_key'] = raion_matches

        # Get active raions
        if active_raions is None:
            raion_counts = df['raion_key'].value_counts()
            active_raions = raion_counts[raion_counts >= 10].index.tolist()
            active_raions = [r for r in active_raions if r is not None]

        print(f"  Matched raions: {len(active_raions)}")

        # Create date range
        n_days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(n_days)]
        date_to_idx = {d.date(): i for i, d in enumerate(dates)}

        # Create raion index
        raion_to_idx = {r: i for i, r in enumerate(active_raions)}
        n_raions = len(active_raions)

        # Initialize with 30 features
        features = np.zeros((n_days, n_raions, self.N_FEATURES), dtype=np.float32)
        mask = np.zeros((n_days, n_raions), dtype=bool)

        # Temporary storage for aggregation with full alert info
        day_raion_alerts: Dict[Tuple[int, int], List[Dict]] = {}

        for _, row in df.iterrows():
            raion_key = row.get('raion_key')
            if raion_key is None or raion_key not in raion_to_idx:
                continue

            date_key = row.get('date')
            if date_key is None or date_key not in date_to_idx:
                continue

            day_idx = date_to_idx[date_key]
            raion_idx = raion_to_idx[raion_key]
            key = (day_idx, raion_idx)

            if key not in day_raion_alerts:
                day_raion_alerts[key] = []

            day_raion_alerts[key].append({
                'duration': row.get('duration_minutes', 0) or 0,
                'hour': row.get('hour', 12),
                'source': row.get('source', 'official'),
                'level': row.get('level', 'raion'),
                'started_at': row.get('started_at'),
            })

        # Aggregate features for each day-raion pair
        for (day_idx, raion_idx), alerts in day_raion_alerts.items():
            mask[day_idx, raion_idx] = True
            feat = features[day_idx, raion_idx]

            durations = [a['duration'] for a in alerts if a['duration'] > 0]
            hours = [a['hour'] for a in alerts]
            sources = [a['source'] for a in alerts]
            levels = [a['level'] for a in alerts]

            # Alert counts [0-3]
            feat[self.FEATURE_IDX['alert_count']] = len(alerts)
            feat[self.FEATURE_IDX['official_alerts']] = sum(1 for s in sources if s == 'official')
            feat[self.FEATURE_IDX['volunteer_alerts']] = sum(1 for s in sources if s == 'volunteer')
            feat[self.FEATURE_IDX['hromada_level_alerts']] = sum(1 for l in levels if l == 'hromada')

            # Duration stats [4-8]
            if durations:
                feat[self.FEATURE_IDX['total_duration_minutes']] = sum(durations)
                feat[self.FEATURE_IDX['avg_duration_minutes']] = np.mean(durations)
                feat[self.FEATURE_IDX['max_duration_minutes']] = max(durations)
                feat[self.FEATURE_IDX['min_duration_minutes']] = min(durations)
                feat[self.FEATURE_IDX['std_duration_minutes']] = np.std(durations) if len(durations) > 1 else 0

            # Time of day [9-12]
            feat[self.FEATURE_IDX['night_alerts']] = sum(1 for h in hours if h >= 22 or h < 6)
            feat[self.FEATURE_IDX['morning_alerts']] = sum(1 for h in hours if 6 <= h < 12)
            feat[self.FEATURE_IDX['afternoon_alerts']] = sum(1 for h in hours if 12 <= h < 18)
            feat[self.FEATURE_IDX['evening_alerts']] = sum(1 for h in hours if 18 <= h < 22)

            # Duration categories [13-16]
            if durations:
                feat[self.FEATURE_IDX['short_alerts']] = sum(1 for d in durations if d < 15)
                feat[self.FEATURE_IDX['medium_alerts']] = sum(1 for d in durations if 15 <= d < 60)
                feat[self.FEATURE_IDX['long_alerts']] = sum(1 for d in durations if 60 <= d < 120)
                feat[self.FEATURE_IDX['extended_alerts']] = sum(1 for d in durations if d >= 120)

            # Temporal sequences [17-20]
            if hours:
                feat[self.FEATURE_IDX['first_alert_hour']] = min(hours)
                feat[self.FEATURE_IDX['last_alert_hour']] = max(hours)

            # Calculate average gap between alerts
            if len(alerts) > 1:
                timestamps = sorted([a['started_at'] for a in alerts if a['started_at'] is not None])
                if len(timestamps) > 1:
                    gaps = []
                    for i in range(1, len(timestamps)):
                        gap = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60
                        gaps.append(gap)
                    if gaps:
                        feat[self.FEATURE_IDX['alert_gap_avg_minutes']] = np.mean(gaps)

            # Intensity metrics [21-24]
            if hours:
                # Alerts per hour (active hours)
                unique_hours = len(set(hours))
                if unique_hours > 0:
                    feat[self.FEATURE_IDX['alerts_per_hour']] = len(alerts) / unique_hours

                # Peak hour density (max alerts in any single hour)
                from collections import Counter
                hour_counts = Counter(hours)
                if hour_counts:
                    feat[self.FEATURE_IDX['peak_hour_density']] = max(hour_counts.values())

            # Cross-raion context [25-27]
            feat[self.FEATURE_IDX['oblast_wide_alert']] = sum(1 for l in levels if l == 'oblast')
            feat[self.FEATURE_IDX['raion_specific']] = sum(1 for l in levels if l == 'raion')

        # Compute derived features and cross-day metrics
        for raion_idx in range(n_raions):
            # Calculate consecutive days with alerts [20]
            consecutive = 0
            max_consecutive = 0
            for day_idx in range(n_days):
                if mask[day_idx, raion_idx]:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            # Store in the last observed day for this raion
            for day_idx in range(n_days - 1, -1, -1):
                if mask[day_idx, raion_idx]:
                    features[day_idx, raion_idx, self.FEATURE_IDX['consecutive_days']] = max_consecutive
                    break

            # Alert frequency change [28]
            for day_idx in range(1, n_days):
                prev_count = features[day_idx - 1, raion_idx, self.FEATURE_IDX['alert_count']]
                curr_count = features[day_idx, raion_idx, self.FEATURE_IDX['alert_count']]
                if prev_count > 0:
                    features[day_idx, raion_idx, self.FEATURE_IDX['alert_frequency_change']] = (
                        (curr_count - prev_count) / prev_count
                    )

            # Duration trend [29] (change in average duration)
            for day_idx in range(1, n_days):
                prev_dur = features[day_idx - 1, raion_idx, self.FEATURE_IDX['avg_duration_minutes']]
                curr_dur = features[day_idx, raion_idx, self.FEATURE_IDX['avg_duration_minutes']]
                if prev_dur > 0:
                    features[day_idx, raion_idx, self.FEATURE_IDX['duration_trend']] = (
                        (curr_dur - prev_dur) / prev_dur
                    )

        # Compute spillover indicators [27] - alerts in neighboring raions
        # This requires raion adjacency information which would need boundary data
        # For now, leave as 0 (could be computed with spatial join later)

        print(f"  Air raid sirens: {mask.sum()} raion-day observations, {self.N_FEATURES} features")

        return features, mask, active_raions, dates


# =============================================================================
# UCDP RAION LOADER
# =============================================================================

class UCDPRaionLoader:
    """
    Load UCDP (Uppsala Conflict Data Program) events with raion-level assignment.

    UCDP provides detailed conflict event data with coordinates for lethal violence.

    Output: Per-raion daily feature tensor [n_days, n_raions, 35 features]

    Features (35 total):
    - [0]: event_count
    - [1-4]: Fatality breakdown (total, military_a, military_b, civilian)
    - [5-7]: Fatality estimates (best, high, low)
    - [8-10]: Violence type (state_based, non_state, one_sided)
    - [11-13]: Actor attribution (russia_side_a, ukraine_side_a, other_actor)
    - [14-16]: Certainty (high_clarity, medium_clarity, low_clarity)
    - [17-19]: Location precision (exact, approximate, admin_level)
    - [20-22]: Date precision (exact_date, week_precision, month_precision)
    - [23-25]: Source credibility (multi_source, official_source, media_source)
    - [26-28]: Intensity (mass_casualty, high_intensity, low_intensity)
    - [29-31]: Temporal patterns (morning, afternoon, night)
    - [32-34]: Derived (lethality_rate, civilian_ratio, event_clustering)
    """

    # Feature name to index mapping (35 features)
    FEATURE_NAMES = [
        # Basic [0]
        'event_count',
        # Fatality breakdown [1-4]
        'fatalities_total', 'fatalities_side_a', 'fatalities_side_b', 'fatalities_civilian',
        # Fatality estimates [5-7]
        'fatalities_best', 'fatalities_high', 'fatalities_low',
        # Violence type [8-10]
        'state_based_conflict', 'non_state_conflict', 'one_sided_violence',
        # Actor attribution [11-13]
        'russia_actor', 'ukraine_actor', 'other_actor',
        # Event certainty [14-16]
        'high_clarity', 'medium_clarity', 'low_clarity',
        # Location precision [17-19]
        'location_exact', 'location_approximate', 'location_admin',
        # Date precision [20-22]
        'date_exact', 'date_week', 'date_month',
        # Source credibility [23-25]
        'multi_source', 'official_source', 'media_source',
        # Intensity [26-28]
        'mass_casualty_event', 'high_intensity', 'low_intensity',
        # Temporal [29-31]
        'events_morning', 'events_afternoon', 'events_night',
        # Derived [32-34]
        'lethality_rate', 'civilian_ratio', 'event_clustering',
    ]

    FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
    N_FEATURES = len(FEATURE_NAMES)

    def __init__(
        self,
        raion_manager: Optional[RaionBoundaryManager] = None,
        data_file: Optional[Path] = None,
    ):
        """
        Initialize the UCDP loader.

        Args:
            raion_manager: RaionBoundaryManager instance
            data_file: Path to UCDP events CSV
        """
        self.raion_manager = raion_manager or RaionBoundaryManager()
        self.data_file = data_file or UCDP_EVENTS_FILE
        self._data: Optional[pd.DataFrame] = None

    def load_raw(self) -> pd.DataFrame:
        """Load raw UCDP data into DataFrame."""
        if self._data is not None:
            return self._data

        if not self.data_file.exists():
            warnings.warn(f"UCDP data not found: {self.data_file}")
            return pd.DataFrame()

        print(f"Loading UCDP data from {self.data_file}...")
        df = pd.read_csv(self.data_file)

        # Filter to Ukraine (country_id = 369 or country = "Ukraine")
        if 'country' in df.columns:
            df = df[df['country'].str.contains('Ukraine', case=False, na=False)]
        elif 'country_id' in df.columns:
            df = df[df['country_id'] == 369]

        # Parse dates
        if 'date_start' in df.columns:
            df['date'] = pd.to_datetime(df['date_start'], errors='coerce')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Filter to valid coordinates
        df = df.dropna(subset=['latitude', 'longitude', 'date'])

        print(f"  Loaded {len(df)} UCDP events for Ukraine")
        self._data = df
        return df

    def load_raion_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        active_raions: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[datetime]]:
        """
        Load UCDP events as per-raion daily features.

        Extracts 35 decomposed features for comprehensive conflict analysis.

        Args:
            start_date: Start of date range
            end_date: End of date range
            active_raions: List of raion keys to include

        Returns:
            Tuple of (features[35], mask, raion_keys, dates)
        """
        start = start_date or DEFAULT_START_DATE
        end = end_date or DEFAULT_END_DATE

        # Load data
        df = self.load_raw()
        if df.empty:
            return np.array([]), np.array([]), [], []

        # Filter date range
        df = df[(df['date'] >= start) & (df['date'] <= end)].copy()

        # Ensure raion manager is loaded
        self.raion_manager.load()

        # Assign raions
        print("  Assigning UCDP events to raions...")
        raion_assignments = []
        for _, row in df.iterrows():
            raion = self.raion_manager.get_raion_for_point(row['latitude'], row['longitude'])
            raion_assignments.append(raion)
        df['raion'] = raion_assignments

        # Get active raions
        if active_raions is None:
            raion_counts = df['raion'].value_counts()
            active_raions = raion_counts[raion_counts >= 5].index.tolist()
            active_raions = [r for r in active_raions if r is not None]

        print(f"  Active raions: {len(active_raions)}")

        # Create date range
        n_days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(n_days)]
        date_to_idx = {d.date(): i for i, d in enumerate(dates)}

        # Create raion index
        raion_to_idx = {r: i for i, r in enumerate(active_raions)}
        n_raions = len(active_raions)

        # Initialize with 35 features
        features = np.zeros((n_days, n_raions, self.N_FEATURES), dtype=np.float32)
        mask = np.zeros((n_days, n_raions), dtype=bool)

        for _, row in df.iterrows():
            raion = row.get('raion')
            if raion is None or raion not in raion_to_idx:
                continue

            date_key = row['date'].date()
            if date_key not in date_to_idx:
                continue

            day_idx = date_to_idx[date_key]
            raion_idx = raion_to_idx[raion]
            feat = features[day_idx, raion_idx]

            mask[day_idx, raion_idx] = True

            # Event count [0]
            feat[self.FEATURE_IDX['event_count']] += 1

            # Fatality breakdown [1-4]
            deaths_a = row.get('deaths_a', 0) or 0
            deaths_b = row.get('deaths_b', 0) or 0
            deaths_civ = row.get('deaths_civilians', 0) or 0
            deaths_unknown = row.get('deaths_unknown', 0) or 0
            total = deaths_a + deaths_b + deaths_civ + deaths_unknown

            feat[self.FEATURE_IDX['fatalities_total']] += total
            feat[self.FEATURE_IDX['fatalities_side_a']] += deaths_a
            feat[self.FEATURE_IDX['fatalities_side_b']] += deaths_b
            feat[self.FEATURE_IDX['fatalities_civilian']] += deaths_civ

            # Fatality estimates [5-7]
            feat[self.FEATURE_IDX['fatalities_best']] += row.get('best', 0) or 0
            feat[self.FEATURE_IDX['fatalities_high']] += row.get('high', 0) or 0
            feat[self.FEATURE_IDX['fatalities_low']] += row.get('low', 0) or 0

            # Violence type [8-10]
            violence_type = row.get('type_of_violence', 0)
            if violence_type == 1:
                feat[self.FEATURE_IDX['state_based_conflict']] += 1
            elif violence_type == 2:
                feat[self.FEATURE_IDX['non_state_conflict']] += 1
            elif violence_type == 3:
                feat[self.FEATURE_IDX['one_sided_violence']] += 1

            # Actor attribution [11-13]
            side_a = str(row.get('side_a', '')).lower()
            side_b = str(row.get('side_b', '')).lower()
            dyad = str(row.get('dyad_name', '')).lower()

            if 'russia' in side_a or 'russian' in side_a:
                feat[self.FEATURE_IDX['russia_actor']] += 1
            if 'ukraine' in side_a or 'ukrainian' in side_a:
                feat[self.FEATURE_IDX['ukraine_actor']] += 1
            if 'russia' not in side_a and 'ukraine' not in side_a:
                feat[self.FEATURE_IDX['other_actor']] += 1

            # Event clarity [14-16]
            clarity = row.get('event_clarity', 1)
            if clarity == 1:
                feat[self.FEATURE_IDX['high_clarity']] += 1
            elif clarity == 2:
                feat[self.FEATURE_IDX['medium_clarity']] += 1
            else:
                feat[self.FEATURE_IDX['low_clarity']] += 1

            # Location precision [17-19]
            where_prec = row.get('where_prec', 1)
            if where_prec == 1:
                feat[self.FEATURE_IDX['location_exact']] += 1
            elif where_prec == 2:
                feat[self.FEATURE_IDX['location_approximate']] += 1
            else:
                feat[self.FEATURE_IDX['location_admin']] += 1

            # Date precision [20-22]
            date_prec = row.get('date_prec', 1)
            if date_prec == 1:
                feat[self.FEATURE_IDX['date_exact']] += 1
            elif date_prec == 2:
                feat[self.FEATURE_IDX['date_week']] += 1
            else:
                feat[self.FEATURE_IDX['date_month']] += 1

            # Source credibility [23-25]
            num_sources = row.get('number_of_sources', 1) or 1
            if num_sources > 1:
                feat[self.FEATURE_IDX['multi_source']] += 1

            source_office = row.get('source_office', '')
            if source_office and str(source_office).strip():
                feat[self.FEATURE_IDX['official_source']] += 1

            source_article = row.get('source_article', '')
            if source_article and str(source_article).strip():
                feat[self.FEATURE_IDX['media_source']] += 1

            # Intensity [26-28]
            best_est = row.get('best', 0) or 0
            if best_est >= 10:
                feat[self.FEATURE_IDX['mass_casualty_event']] += 1
            if best_est >= 5:
                feat[self.FEATURE_IDX['high_intensity']] += 1
            else:
                feat[self.FEATURE_IDX['low_intensity']] += 1

        # Compute derived features [32-34]
        for day_idx in range(n_days):
            for raion_idx in range(n_raions):
                feat = features[day_idx, raion_idx]
                events = feat[self.FEATURE_IDX['event_count']]

                if events > 0:
                    total_fatalities = feat[self.FEATURE_IDX['fatalities_total']]
                    civ_fatalities = feat[self.FEATURE_IDX['fatalities_civilian']]

                    # Lethality rate (fatalities per event)
                    feat[self.FEATURE_IDX['lethality_rate']] = total_fatalities / events

                    # Civilian ratio
                    if total_fatalities > 0:
                        feat[self.FEATURE_IDX['civilian_ratio']] = civ_fatalities / total_fatalities

        # Compute event clustering [34]
        for raion_idx in range(n_raions):
            for day_idx in range(n_days):
                window_start = max(0, day_idx - 3)
                window_end = min(n_days, day_idx + 4)
                cluster_count = sum(
                    features[d, raion_idx, self.FEATURE_IDX['event_count']]
                    for d in range(window_start, window_end)
                    if d != day_idx
                )
                features[day_idx, raion_idx, self.FEATURE_IDX['event_clustering']] = cluster_count

        print(f"  UCDP: {mask.sum()} raion-day observations, {self.N_FEATURES} features")

        return features, mask, active_raions, dates


# =============================================================================
# WARSPOTTING RAION LOADER
# =============================================================================

# Warspotting data file
WARSPOTTING_FILE = DATA_DIR / "warspotting" / "warspotting_losses_api.json"


class WarspottingRaionLoader:
    """
    Load Warspotting equipment loss data with raion-level assignment.

    Warspotting provides geolocated equipment losses with:
    - 59.5% have direct geo coordinates
    - 90.2% have nearest_location (parseable raion names)

    Output: Per-raion daily feature tensor [n_days, n_raions, 33 features]

    Features (33 total):
    - [0]: total_losses
    - [1-4]: Status breakdown (destroyed, captured, abandoned, damaged)
    - [5-6]: Side attribution (russian_losses, ukrainian_losses)
    - [7-16]: Equipment categories (tanks, ifvs, apcs, artillery_towed,
              artillery_sp, mlrs, air_defense, aircraft, helicopters, drones)
    - [17-21]: Vehicles (trucks, engineering, command, medical, other)
    - [22-26]: High-value (modern_tanks, modern_ifvs, awacs_ew, strategic_sam, cruise_launchers)
    - [27-30]: Tags (cope_cage, era_equipped, atgm_equipped, drone_modified)
    - [31-32]: Derived (loss_rate_change, destroyed_captured_ratio)
    """

    # Feature name to index mapping (33 features)
    FEATURE_NAMES = [
        # Basic [0]
        'total_losses',
        # Status [1-4]
        'destroyed', 'captured', 'abandoned', 'damaged',
        # Side [5-6]
        'russian_losses', 'ukrainian_losses',
        # Heavy equipment [7-11]
        'tanks', 'ifvs', 'apcs', 'artillery_towed', 'artillery_sp',
        # Fire support [12-14]
        'mlrs', 'air_defense', 'aircraft',
        # Air [15-16]
        'helicopters', 'drones',
        # Vehicles [17-21]
        'trucks', 'engineering', 'command_vehicles', 'medical', 'other_equipment',
        # High-value [22-26]
        'modern_tanks', 'modern_ifvs', 'awacs_ew', 'strategic_sam', 'cruise_launchers',
        # Tags/mods [27-30]
        'cope_cage', 'era_equipped', 'atgm_equipped', 'drone_modified',
        # Derived [31-32]
        'loss_rate_change', 'destroyed_captured_ratio',
    ]

    FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
    N_FEATURES = len(FEATURE_NAMES)

    # Equipment type patterns (expanded)
    EQUIPMENT_TYPES = {
        'tanks': ['tank', 't-72', 't-80', 't-90', 't-62', 't-64', 't-55'],
        'ifvs': ['bmp', 'bmd', 'ifv', 'infantry fighting', 'bradley', 'marder'],
        'apcs': ['btr', 'apc', 'mtlb', 'mt-lb', 'mrap', 'mastiff', 'spartan'],
        'artillery_towed': ['towed artillery', 'd-30', 'd-20', 'm-46', '2a65', 'm777'],
        'artillery_sp': ['self-propelled', '2s19', '2s1', '2s3', '2s5', 'msta', 'akatsiya', 'pzh'],
        'mlrs': ['mlrs', 'grad', 'bm-21', 'bm-27', 'smerch', 'tornado', 'himars', 'uragan'],
        'air_defense': ['air defense', 'sam', 'tor', 'buk', 's-300', 's-400', 'pantsir', 'tunguska', 'osa', 'strela'],
        'aircraft': ['airplane', 'aircraft', 'jet', 'fighter', 'bomber', 'su-', 'mig-', 'il-', 'a-50'],
        'helicopters': ['helicopter', 'mi-', 'ka-'],
        'drones': ['drone', 'uav', 'orlan', 'forpost', 'zala', 'supercam', 'bayraktar'],
        'trucks': ['truck', 'kamaz', 'ural', 'transport', 'supply'],
        'engineering': ['engineering', 'imr', 'brem', 'recovery', 'bridging', 'mtk', 'mtu'],
        'command_vehicles': ['command', 'communications', 'r-145', 'r-149', 'p-260'],
        'medical': ['ambulance', 'medical', 'санитар'],
    }

    # Modern/high-value equipment patterns
    HIGH_VALUE_PATTERNS = {
        'modern_tanks': ['t-90m', 't-80bvm', 't-72b3', 'leopard', 'challenger', 'abrams'],
        'modern_ifvs': ['bmp-3', 'bmpt', 'bradley', 'cv90', 'marder'],
        'awacs_ew': ['a-50', 'beriev', 'il-22', 'krasukha', 'zhitel', 'borisoglebsk'],
        'strategic_sam': ['s-300', 's-400', 's-350', 'patriot'],
        'cruise_launchers': ['iskander', 'kalibr', 'kinzhal', 'bastion'],
    }

    # Tag patterns
    TAG_PATTERNS = {
        'cope_cage': ['cope cage', 'cope-cage', 'anti-drone cage', 'roof cage'],
        'era_equipped': ['era ', 'kontakt', 'relikt', 'reactive armor'],
        'atgm_equipped': ['atgm', 'kornet', 'metis', 'konkurs', 'shturm'],
        'drone_modified': ['drone carrier', 'fpv rack', 'drone launcher'],
    }

    def __init__(
        self,
        raion_manager: Optional[RaionBoundaryManager] = None,
        data_file: Optional[Path] = None,
    ):
        self.raion_manager = raion_manager or RaionBoundaryManager()
        self.data_file = data_file or WARSPOTTING_FILE
        self._data: Optional[pd.DataFrame] = None
        self._raion_name_map: Optional[Dict[str, str]] = None

    def _build_raion_name_map(self) -> Dict[str, str]:
        """Build mapping from location names to raion keys."""
        if self._raion_name_map is not None:
            return self._raion_name_map

        self.raion_manager.load()

        name_map = {}
        for key, raion in self.raion_manager.raions.items():
            # Normalize names
            oblast_norm = raion.oblast.lower().replace("'", "").replace("oblast", "").strip()
            raion_norm = raion.name.lower().replace("'", "").replace("raion", "").strip()

            name_map[(oblast_norm, raion_norm)] = key
            name_map[raion_norm] = key  # Also allow just raion name

        self._raion_name_map = name_map
        return name_map

    def _parse_nearest_location(self, location: str) -> Optional[str]:
        """Parse nearest_location string to extract raion key."""
        if not location:
            return None

        name_map = self._build_raion_name_map()
        location_lower = location.lower()

        # Try to extract raion name from formats like:
        # "Novotroitske (Pokrovsk hromada), Pokrovsk raion"
        # "Pokrovsk raion, Donetsk oblast"

        # Look for "X raion" pattern
        import re
        raion_match = re.search(r'(\w+)\s+raion', location_lower)
        if raion_match:
            raion_name = raion_match.group(1).strip()
            if raion_name in name_map:
                return name_map[raion_name]

        # Try partial matching
        for raion_norm, key in name_map.items():
            if isinstance(raion_norm, str) and raion_norm in location_lower:
                return key

        return None

    def _classify_equipment(self, type_str: str, model_str: str) -> str:
        """Classify equipment from type and model strings."""
        combined = f"{type_str or ''} {model_str or ''}".lower()

        for category, patterns in self.EQUIPMENT_TYPES.items():
            if any(p in combined for p in patterns):
                return category
        return 'other'

    def load_raw(self) -> pd.DataFrame:
        """Load raw Warspotting data into DataFrame."""
        if self._data is not None:
            return self._data

        if not self.data_file.exists():
            warnings.warn(f"Warspotting data not found: {self.data_file}")
            return pd.DataFrame()

        print(f"Loading Warspotting data from {self.data_file}...")

        with open(self.data_file) as f:
            data = json.load(f)

        losses = data.get('losses', [])

        records = []
        for item in losses:
            geo = item.get('geo')
            lat, lon = None, None

            # Parse geo string "lat,lon"
            if geo and isinstance(geo, str) and ',' in geo:
                try:
                    parts = geo.split(',')
                    lat = float(parts[0])
                    lon = float(parts[1])
                except (ValueError, IndexError):
                    pass

            records.append({
                'id': item.get('id'),
                'type': item.get('type', ''),
                'model': item.get('model', ''),
                'status': item.get('status', ''),
                'date': item.get('date'),
                'lat': lat,
                'lon': lon,
                'nearest_location': item.get('nearest_location', ''),
                'unit': item.get('unit'),
                'tags': item.get('tags'),
            })

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        print(f"  Loaded {len(df)} Warspotting losses")
        print(f"  With geo coords: {df['lat'].notna().sum()}")
        print(f"  With nearest_location: {(df['nearest_location'] != '').sum()}")

        self._data = df
        return df

    def _classify_high_value(self, type_str: str, model_str: str) -> Optional[str]:
        """Check if equipment is high-value and return category."""
        text = f"{type_str or ''} {model_str or ''}".lower()
        for category, patterns in self.HIGH_VALUE_PATTERNS.items():
            if any(p in text for p in patterns):
                return category
        return None

    def _extract_tags(self, tags_str: str) -> List[str]:
        """Extract tag categories from tags field."""
        if not tags_str:
            return []
        tags_lower = tags_str.lower()
        found = []
        for tag_name, patterns in self.TAG_PATTERNS.items():
            if any(p in tags_lower for p in patterns):
                found.append(tag_name)
        return found

    def load_raion_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        active_raions: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[datetime]]:
        """
        Load Warspotting losses as per-raion daily features.

        Extracts 33 decomposed features for comprehensive analysis.
        Uses geo coordinates when available, falls back to nearest_location parsing.
        """
        start = start_date or DEFAULT_START_DATE
        end = end_date or DEFAULT_END_DATE

        df = self.load_raw()
        if df.empty:
            return np.array([]), np.array([]), [], []

        df = df[(df['date'] >= start) & (df['date'] <= end)].copy()

        self.raion_manager.load()

        # Assign raions - prefer geo, fall back to nearest_location
        print("  Assigning Warspotting losses to raions...")
        raion_assignments = []
        geo_assigned = 0
        name_assigned = 0

        for _, row in df.iterrows():
            raion = None

            # Try geo coordinates first
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                raion = self.raion_manager.get_raion_for_point(row['lat'], row['lon'])
                if raion:
                    geo_assigned += 1

            # Fall back to nearest_location parsing
            if raion is None and row.get('nearest_location'):
                raion = self._parse_nearest_location(row['nearest_location'])
                if raion:
                    name_assigned += 1

            raion_assignments.append(raion)

        df['raion'] = raion_assignments
        print(f"    Assigned via geo: {geo_assigned}")
        print(f"    Assigned via name: {name_assigned}")
        print(f"    Unassigned: {len(df) - geo_assigned - name_assigned}")

        # Get active raions
        if active_raions is None:
            raion_counts = df['raion'].value_counts()
            active_raions = raion_counts[raion_counts >= 5].index.tolist()
            active_raions = [r for r in active_raions if r is not None]

        print(f"  Active raions: {len(active_raions)}")

        # Create date range
        n_days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(n_days)]
        date_to_idx = {d.date(): i for i, d in enumerate(dates)}

        raion_to_idx = {r: i for i, r in enumerate(active_raions)}
        n_raions = len(active_raions)

        # Initialize with 33 features
        features = np.zeros((n_days, n_raions, self.N_FEATURES), dtype=np.float32)
        mask = np.zeros((n_days, n_raions), dtype=bool)

        # Equipment type to feature index mapping
        equipment_idx = {
            'tanks': self.FEATURE_IDX['tanks'],
            'ifvs': self.FEATURE_IDX['ifvs'],
            'apcs': self.FEATURE_IDX['apcs'],
            'artillery_towed': self.FEATURE_IDX['artillery_towed'],
            'artillery_sp': self.FEATURE_IDX['artillery_sp'],
            'mlrs': self.FEATURE_IDX['mlrs'],
            'air_defense': self.FEATURE_IDX['air_defense'],
            'aircraft': self.FEATURE_IDX['aircraft'],
            'helicopters': self.FEATURE_IDX['helicopters'],
            'drones': self.FEATURE_IDX['drones'],
            'trucks': self.FEATURE_IDX['trucks'],
            'engineering': self.FEATURE_IDX['engineering'],
            'command_vehicles': self.FEATURE_IDX['command_vehicles'],
            'medical': self.FEATURE_IDX['medical'],
        }

        # Status to feature index
        status_idx = {
            'Destroyed': self.FEATURE_IDX['destroyed'],
            'Captured': self.FEATURE_IDX['captured'],
            'Abandoned': self.FEATURE_IDX['abandoned'],
            'Damaged': self.FEATURE_IDX['damaged'],
        }

        # High-value to feature index
        hv_idx = {
            'modern_tanks': self.FEATURE_IDX['modern_tanks'],
            'modern_ifvs': self.FEATURE_IDX['modern_ifvs'],
            'awacs_ew': self.FEATURE_IDX['awacs_ew'],
            'strategic_sam': self.FEATURE_IDX['strategic_sam'],
            'cruise_launchers': self.FEATURE_IDX['cruise_launchers'],
        }

        # Tag to feature index
        tag_idx = {
            'cope_cage': self.FEATURE_IDX['cope_cage'],
            'era_equipped': self.FEATURE_IDX['era_equipped'],
            'atgm_equipped': self.FEATURE_IDX['atgm_equipped'],
            'drone_modified': self.FEATURE_IDX['drone_modified'],
        }

        for _, row in df.iterrows():
            raion = row.get('raion')
            if raion is None or raion not in raion_to_idx:
                continue

            date_key = row['date'].date() if pd.notna(row['date']) else None
            if date_key is None or date_key not in date_to_idx:
                continue

            day_idx = date_to_idx[date_key]
            raion_idx = raion_to_idx[raion]
            feat = features[day_idx, raion_idx]

            mask[day_idx, raion_idx] = True

            # Total losses [0]
            feat[self.FEATURE_IDX['total_losses']] += 1

            # Status breakdown [1-4]
            status = row.get('status', 'Destroyed')
            if status in status_idx:
                feat[status_idx[status]] += 1

            # Side attribution [5-6]
            lost_by = row.get('lost_by', '')
            if lost_by == 'Russia':
                feat[self.FEATURE_IDX['russian_losses']] += 1
            elif lost_by == 'Ukraine':
                feat[self.FEATURE_IDX['ukrainian_losses']] += 1

            # Equipment classification [7-21]
            type_str = row.get('type', '')
            model_str = row.get('model', '')
            eq_type = self._classify_equipment(type_str, model_str)
            if eq_type in equipment_idx:
                feat[equipment_idx[eq_type]] += 1
            else:
                feat[self.FEATURE_IDX['other_equipment']] += 1

            # High-value targets [22-26]
            hv_type = self._classify_high_value(type_str, model_str)
            if hv_type and hv_type in hv_idx:
                feat[hv_idx[hv_type]] += 1

            # Tags/modifications [27-30]
            tags = row.get('tags', '')
            for tag_name in self._extract_tags(tags):
                if tag_name in tag_idx:
                    feat[tag_idx[tag_name]] += 1

        # Compute derived features [31-32]
        for day_idx in range(n_days):
            for raion_idx in range(n_raions):
                feat = features[day_idx, raion_idx]
                total = feat[self.FEATURE_IDX['total_losses']]

                if total > 0:
                    destroyed = feat[self.FEATURE_IDX['destroyed']]
                    captured = feat[self.FEATURE_IDX['captured']]

                    # Destroyed/captured ratio (captures are more impactful)
                    if captured > 0:
                        feat[self.FEATURE_IDX['destroyed_captured_ratio']] = destroyed / captured
                    else:
                        feat[self.FEATURE_IDX['destroyed_captured_ratio']] = destroyed

        # Compute loss rate change [31]
        for raion_idx in range(n_raions):
            for day_idx in range(1, n_days):
                prev_total = features[day_idx - 1, raion_idx, self.FEATURE_IDX['total_losses']]
                curr_total = features[day_idx, raion_idx, self.FEATURE_IDX['total_losses']]
                if prev_total > 0:
                    features[day_idx, raion_idx, self.FEATURE_IDX['loss_rate_change']] = (
                        (curr_total - prev_total) / prev_total
                    )

        print(f"  Warspotting: {mask.sum()} raion-day observations, {self.N_FEATURES} features")

        return features, mask, active_raions, dates


# =============================================================================
# DEEPSTATE RAION LOADER
# =============================================================================

# Import DeepState constants and helpers
try:
    from .deepstate_spatial_loader import (
        UNIT_PATTERNS,
        ATTACK_DIRECTION_PATTERNS,
        classify_unit_type,
        is_military_unit,
        is_attack_direction,
        parse_wayback_timestamp,
    )
    HAS_DEEPSTATE_HELPERS = True
except ImportError:
    HAS_DEEPSTATE_HELPERS = False
    UNIT_PATTERNS = []
    ATTACK_DIRECTION_PATTERNS = []

# DeepState data paths
DEEPSTATE_WAYBACK_DIR = DATA_DIR / "deepstate" / "wayback_snapshots"


class DeepStateRaionLoader:
    """
    Load DeepState data with raion-level assignment.

    DeepState provides military unit positions, frontline boundaries, and attack
    direction markers. This loader assigns Point features (units, attack directions)
    to raions via point-in-polygon, and computes polygon intersection metrics
    for frontline boundaries.

    Output: Per-raion daily feature tensor [n_days, n_raions, 48 features]

    Features (48 total):
    - [0-3]: Unit counts (total, russian, ukrainian, unknown_side)
    - [4-10]: Unit types (infantry, armor, artillery, airborne, naval, pmc, support)
    - [11-14]: Unit echelons (brigade, regiment, battalion, company)
    - [15-19]: Attack directions (total, north, south, east, west)
    - [20-23]: Control state (liberated, contested, occupied, front_line)
    - [24-27]: Territorial metrics (area_km2, perimeter_km, centroid_shift, boundary_change)
    - [28-31]: Infrastructure (airports, headquarters, logistics, fortifications)
    - [32-35]: Movement indicators (advancing, retreating, static, rotation)
    - [36-39]: Intensity (combat_active, shelling_zone, breakthrough, encirclement)
    - [40-43]: Temporal change (units_delta, attacks_delta, area_delta, control_delta)
    - [44-47]: Derived (unit_density, attack_concentration, front_stability, activity_level)
    """

    # Feature name to index mapping (48 features)
    FEATURE_NAMES = [
        # Unit counts [0-3]
        'unit_count', 'russian_units', 'ukrainian_units', 'unknown_side_units',
        # Unit types [4-10]
        'unit_infantry', 'unit_armor', 'unit_artillery', 'unit_airborne',
        'unit_naval', 'unit_pmc', 'unit_support',
        # Unit echelons [11-14]
        'echelon_brigade', 'echelon_regiment', 'echelon_battalion', 'echelon_company',
        # Attack directions [15-19]
        'attack_directions', 'attacks_north', 'attacks_south', 'attacks_east', 'attacks_west',
        # Control state [20-23]
        'territory_liberated', 'territory_contested', 'territory_occupied', 'front_line_present',
        # Territorial metrics [24-27]
        'territorial_area_km2', 'perimeter_km', 'centroid_shift_km', 'boundary_change_km',
        # Infrastructure [28-31]
        'airports', 'headquarters', 'logistics_points', 'fortifications',
        # Movement indicators [32-35]
        'units_advancing', 'units_retreating', 'units_static', 'unit_rotation',
        # Combat intensity [36-39]
        'combat_active', 'shelling_zone', 'breakthrough_attempt', 'encirclement_risk',
        # Temporal change [40-43]
        'units_delta', 'attacks_delta', 'area_delta', 'control_delta',
        # Derived [44-47]
        'unit_density', 'attack_concentration', 'front_stability', 'activity_level',
    ]

    FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
    N_FEATURES = len(FEATURE_NAMES)

    # Unit type classification patterns (expanded)
    UNIT_TYPE_PATTERNS = {
        'infantry': [r'бригад', r'полк', r'батальйон', r'піхот', r'мотострілк', r'стрілець'],
        'armor': [r'танков', r'механіз', r'бронет', r'танкова'],
        'artillery': [r'артилер', r'гаубиц', r'ракетн', r'реактивн', r'гармат'],
        'airborne': [r'десант', r'повітрян', r'вдв', r'парашут'],
        'naval': [r'морськ', r'флот', r'берегов'],
        'pmc': [r'ЧВК', r'Wagner', r'Вагнер', r'ПВК', r'приватн'],
        'support': [r'інженер', r'зв\'язк', r'логіст', r'тилов', r'ремонт'],
    }

    # Echelon patterns
    ECHELON_PATTERNS = {
        'brigade': [r'бригад', r'brigade'],
        'regiment': [r'полк', r'regiment'],
        'battalion': [r'батальйон', r'battalion', r'бтгр'],
        'company': [r'рот[аи]', r'company', r'взвод'],
    }

    # Control state patterns (from polygon fill colors)
    CONTROL_COLORS = {
        'liberated': ['#00ff00', '#00cc00', '#green', 'liberated'],
        'contested': ['#ffff00', '#orange', '#ffa500', 'contested', 'gray', 'grey'],
        'occupied': ['#ff0000', '#cc0000', '#red', 'occupied'],
    }

    # Infrastructure patterns
    INFRASTRUCTURE_PATTERNS = {
        'airports': [r'аеропорт', r'аеродром', r'airport', r'airfield', r'летовищ'],
        'headquarters': [r'штаб', r'командн', r'headquarter', r'command'],
        'logistics': [r'логіст', r'склад', r'depot', r'supply', r'постачанн'],
        'fortifications': [r'укріплен', r'фортифік', r'defense', r'оборон', r'позиці'],
    }

    def __init__(
        self,
        raion_manager: Optional[RaionBoundaryManager] = None,
        wayback_dir: Optional[Path] = None,
        latest_per_day: bool = True,
    ):
        """
        Initialize the DeepState raion loader.

        Args:
            raion_manager: RaionBoundaryManager instance
            wayback_dir: Directory containing wayback JSON snapshots
            latest_per_day: If True, use only latest snapshot per day
        """
        self.raion_manager = raion_manager or RaionBoundaryManager()
        self.wayback_dir = wayback_dir or DEEPSTATE_WAYBACK_DIR
        self.latest_per_day = latest_per_day
        self._snapshot_index: Optional[pd.DataFrame] = None

    def _build_snapshot_index(self) -> pd.DataFrame:
        """Build index of available snapshots with timestamps."""
        if self._snapshot_index is not None:
            return self._snapshot_index

        records = []
        for json_file in self.wayback_dir.glob("deepstate_wayback_*.json"):
            if HAS_DEEPSTATE_HELPERS:
                ts = parse_wayback_timestamp(json_file.name)
            else:
                # Fallback parsing
                import re
                match = re.search(r'(\d{14})', json_file.name)
                if match:
                    try:
                        ts = datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
                    except ValueError:
                        ts = None
                else:
                    ts = None

            if ts:
                records.append({
                    'file': json_file,
                    'timestamp': ts,
                    'date': ts.date(),
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('timestamp')
            if self.latest_per_day:
                # Keep only latest snapshot per day
                df = df.groupby('date').last().reset_index()
                df = df.sort_values('timestamp')

        self._snapshot_index = df
        return df

    def _parse_snapshot(self, json_file: Path) -> Dict[str, Any]:
        """Parse a single wayback snapshot file."""
        with open(json_file) as f:
            data = json.load(f)

        # Extract features from nested structure
        map_data = data.get('map', {})
        features = map_data.get('features', [])

        return {
            'features': features,
            'timestamp': data.get('datetime', ''),
            'metadata': data.get('_wayback_metadata', {}),
        }

    def _classify_unit_type(self, name: str, description: str = '') -> str:
        """Classify a military unit by type based on name/description."""
        import re
        text = f"{name} {description}".lower()

        for unit_type, patterns in self.UNIT_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return unit_type

        return 'other'

    def _is_unit_feature(self, feature: Dict) -> bool:
        """Check if a feature represents a military unit."""
        props = feature.get('properties', {})
        name = props.get('name', '')

        if HAS_DEEPSTATE_HELPERS:
            return is_military_unit(name)

        # Fallback check
        import re
        unit_patterns = [
            r'\d+.*бригад', r'\d+.*полк', r'\d+.*дивіз',
            r'\d+.*батальйон', r'ЧВК', r'Wagner',
        ]
        for pattern in unit_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return True
        return False

    def _is_attack_direction(self, feature: Dict) -> bool:
        """Check if a feature represents an attack direction marker."""
        props = feature.get('properties', {})
        name = props.get('name', '')

        if HAS_DEEPSTATE_HELPERS:
            return is_attack_direction(name)

        # Fallback check
        import re
        patterns = [r'напрямок.*атак', r'атак.*напрямок', r'arrow']
        for pattern in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return True
        return False

    def _classify_echelon(self, name: str, description: str = '') -> str:
        """Classify unit echelon level."""
        import re
        text = f"{name} {description}".lower()

        for echelon, patterns in self.ECHELON_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return echelon
        return 'battalion'  # Default echelon

    def _classify_infrastructure(self, name: str, description: str = '') -> Optional[str]:
        """Check if feature is infrastructure and return type."""
        import re
        text = f"{name} {description}".lower()

        for infra_type, patterns in self.INFRASTRUCTURE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return infra_type
        return None

    def _extract_attack_direction(self, name: str, icon: str = '') -> Optional[str]:
        """Extract attack direction (north/south/east/west) from feature."""
        text = f"{name} {icon}".lower()

        if any(d in text for d in ['північ', 'north', 'пн', 'n_']):
            return 'north'
        if any(d in text for d in ['південь', 'south', 'пд', 's_']):
            return 'south'
        if any(d in text for d in ['схід', 'east', 'сх', 'e_']):
            return 'east'
        if any(d in text for d in ['захід', 'west', 'зх', 'w_']):
            return 'west'
        return None

    def _classify_control_state(self, props: Dict, fill_color: str = '') -> str:
        """Classify territorial control state from properties/colors."""
        fill = (props.get('fill', '') or props.get('color', '') or fill_color).lower()

        for state, patterns in self.CONTROL_COLORS.items():
            if any(p in fill for p in patterns):
                return state

        # Check name/description for indicators
        name = (props.get('name', '') or '').lower()
        if 'визволен' in name or 'liberat' in name:
            return 'liberated'
        if 'окупован' in name or 'occupied' in name:
            return 'occupied'

        return 'contested'

    def load_raion_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        active_raions: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[datetime]]:
        """
        Load DeepState features aggregated per raion per day.

        Extracts 48 decomposed features for comprehensive military situation analysis.

        Args:
            start_date: Start of date range
            end_date: End of date range
            active_raions: List of raion keys to include

        Returns:
            Tuple of (features[48], mask, raion_keys, dates)
        """
        start = start_date or DEFAULT_START_DATE
        end = end_date or DEFAULT_END_DATE

        print(f"Loading DeepState data from {self.wayback_dir}...")

        # Build snapshot index
        snapshot_df = self._build_snapshot_index()
        if snapshot_df.empty:
            warnings.warn("No DeepState snapshots found")
            return np.array([]), np.array([]), [], []

        # Filter date range
        snapshot_df = snapshot_df[
            (snapshot_df['date'] >= start.date()) &
            (snapshot_df['date'] <= end.date())
        ]
        print(f"  Found {len(snapshot_df)} snapshots in date range")

        # Load raion boundaries
        self.raion_manager.load()

        # Determine active raions
        if active_raions is None:
            active_raions = list(self.raion_manager.raions.keys())
        active_raions = sorted(active_raions)

        # Create date range
        n_days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(n_days)]
        date_to_idx = {d.date(): i for i, d in enumerate(dates)}

        # Create raion index
        raion_to_idx = {r: i for i, r in enumerate(active_raions)}
        n_raions = len(active_raions)

        # Initialize with 48 features
        features = np.zeros((n_days, n_raions, self.N_FEATURES), dtype=np.float32)
        mask = np.zeros((n_days, n_raions), dtype=bool)

        # Unit type to feature index
        unit_type_idx = {
            'infantry': self.FEATURE_IDX['unit_infantry'],
            'armor': self.FEATURE_IDX['unit_armor'],
            'artillery': self.FEATURE_IDX['unit_artillery'],
            'airborne': self.FEATURE_IDX['unit_airborne'],
            'naval': self.FEATURE_IDX['unit_naval'],
            'pmc': self.FEATURE_IDX['unit_pmc'],
            'support': self.FEATURE_IDX['unit_support'],
            'other': self.FEATURE_IDX['unit_support'],  # Fallback to support
        }

        # Echelon to feature index
        echelon_idx = {
            'brigade': self.FEATURE_IDX['echelon_brigade'],
            'regiment': self.FEATURE_IDX['echelon_regiment'],
            'battalion': self.FEATURE_IDX['echelon_battalion'],
            'company': self.FEATURE_IDX['echelon_company'],
        }

        # Direction to feature index
        direction_idx = {
            'north': self.FEATURE_IDX['attacks_north'],
            'south': self.FEATURE_IDX['attacks_south'],
            'east': self.FEATURE_IDX['attacks_east'],
            'west': self.FEATURE_IDX['attacks_west'],
        }

        # Infrastructure to feature index
        infra_idx = {
            'airports': self.FEATURE_IDX['airports'],
            'headquarters': self.FEATURE_IDX['headquarters'],
            'logistics': self.FEATURE_IDX['logistics_points'],
            'fortifications': self.FEATURE_IDX['fortifications'],
        }

        # Control state to feature index
        control_idx = {
            'liberated': self.FEATURE_IDX['territory_liberated'],
            'contested': self.FEATURE_IDX['territory_contested'],
            'occupied': self.FEATURE_IDX['territory_occupied'],
        }

        # Process each snapshot
        print("  Assigning DeepState features to raions...")
        for _, row in snapshot_df.iterrows():
            day_idx = date_to_idx.get(row['date'])
            if day_idx is None:
                continue

            try:
                snapshot = self._parse_snapshot(row['file'])
            except Exception as e:
                warnings.warn(f"Error parsing {row['file']}: {e}")
                continue

            # Process each feature
            for geojson_feat in snapshot.get('features', []):
                geom = geojson_feat.get('geometry', {})
                geom_type = geom.get('type', '')
                props = geojson_feat.get('properties', {})

                # Handle Point features (units, attack directions, infrastructure)
                if geom_type == 'Point':
                    coords = geom.get('coordinates', [])
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]

                        raion_key = self.raion_manager.get_raion_for_point(lat, lon)
                        if raion_key and raion_key in raion_to_idx:
                            raion_idx = raion_to_idx[raion_key]
                            feat = features[day_idx, raion_idx]
                            mask[day_idx, raion_idx] = True

                            name = props.get('name', '') or ''
                            desc = props.get('description', '') or ''
                            icon = props.get('icon', '') or ''

                            # Unit features
                            if self._is_unit_feature(geojson_feat):
                                feat[self.FEATURE_IDX['unit_count']] += 1

                                # Unit type
                                unit_type = self._classify_unit_type(name, desc)
                                feat[unit_type_idx.get(unit_type, unit_type_idx['other'])] += 1

                                # Echelon
                                echelon = self._classify_echelon(name, desc)
                                feat[echelon_idx.get(echelon, echelon_idx['battalion'])] += 1

                                # Side attribution (from icon path or name)
                                if 'russia' in icon.lower() or 'enemy' in name.lower():
                                    feat[self.FEATURE_IDX['russian_units']] += 1
                                elif 'ukraine' in icon.lower() or 'нашi' in name.lower():
                                    feat[self.FEATURE_IDX['ukrainian_units']] += 1
                                else:
                                    feat[self.FEATURE_IDX['unknown_side_units']] += 1

                            # Attack direction features
                            elif self._is_attack_direction(geojson_feat):
                                feat[self.FEATURE_IDX['attack_directions']] += 1

                                # Direction classification
                                direction = self._extract_attack_direction(name, icon)
                                if direction and direction in direction_idx:
                                    feat[direction_idx[direction]] += 1

                            # Infrastructure features
                            infra_type = self._classify_infrastructure(name, desc)
                            if infra_type and infra_type in infra_idx:
                                feat[infra_idx[infra_type]] += 1

                # Handle Polygon features (territorial control)
                elif geom_type == 'Polygon':
                    try:
                        from shapely.geometry import shape as shapely_shape
                        polygon = shapely_shape(geom)
                        centroid = polygon.centroid

                        raion_key = self.raion_manager.get_raion_for_point(
                            centroid.y, centroid.x
                        )
                        if raion_key and raion_key in raion_to_idx:
                            raion_idx = raion_to_idx[raion_key]
                            feat = features[day_idx, raion_idx]
                            mask[day_idx, raion_idx] = True

                            # Territorial metrics [24-27]
                            area_deg2 = polygon.area
                            area_km2 = area_deg2 * 111 * 73  # Approx at 49°N
                            feat[self.FEATURE_IDX['territorial_area_km2']] += area_km2

                            perim_deg = polygon.length
                            perim_km = perim_deg * 92
                            feat[self.FEATURE_IDX['perimeter_km']] += perim_km

                            # Control state [20-23]
                            fill = props.get('fill', '') or props.get('style', {}).get('fill', '')
                            control_state = self._classify_control_state(props, fill)
                            if control_state in control_idx:
                                feat[control_idx[control_state]] += 1

                            # Front line indicator
                            if 'front' in (props.get('name', '') or '').lower():
                                feat[self.FEATURE_IDX['front_line_present']] += 1

                    except Exception:
                        pass

        # Compute temporal change features [40-43]
        for raion_idx in range(n_raions):
            for day_idx in range(1, n_days):
                # Units delta
                prev_units = features[day_idx - 1, raion_idx, self.FEATURE_IDX['unit_count']]
                curr_units = features[day_idx, raion_idx, self.FEATURE_IDX['unit_count']]
                features[day_idx, raion_idx, self.FEATURE_IDX['units_delta']] = curr_units - prev_units

                # Attacks delta
                prev_attacks = features[day_idx - 1, raion_idx, self.FEATURE_IDX['attack_directions']]
                curr_attacks = features[day_idx, raion_idx, self.FEATURE_IDX['attack_directions']]
                features[day_idx, raion_idx, self.FEATURE_IDX['attacks_delta']] = curr_attacks - prev_attacks

                # Area delta
                prev_area = features[day_idx - 1, raion_idx, self.FEATURE_IDX['territorial_area_km2']]
                curr_area = features[day_idx, raion_idx, self.FEATURE_IDX['territorial_area_km2']]
                features[day_idx, raion_idx, self.FEATURE_IDX['area_delta']] = curr_area - prev_area

        # Compute derived features [44-47]
        for day_idx in range(n_days):
            for raion_idx in range(n_raions):
                feat = features[day_idx, raion_idx]
                area = feat[self.FEATURE_IDX['territorial_area_km2']]
                units = feat[self.FEATURE_IDX['unit_count']]
                attacks = feat[self.FEATURE_IDX['attack_directions']]

                # Unit density (units per 100 km²)
                if area > 0:
                    feat[self.FEATURE_IDX['unit_density']] = (units / area) * 100
                    feat[self.FEATURE_IDX['attack_concentration']] = (attacks / area) * 100

                # Activity level (combined metric)
                feat[self.FEATURE_IDX['activity_level']] = units + attacks * 2

                # Front stability (inverse of territorial changes)
                area_delta = abs(feat[self.FEATURE_IDX['area_delta']]) if day_idx > 0 else 0
                if area > 0:
                    feat[self.FEATURE_IDX['front_stability']] = 1.0 / (1.0 + area_delta / area)

        # Count active raions with observations
        raions_with_data = np.where(mask.sum(axis=0) > 0)[0]
        active_raion_keys = [active_raions[i] for i in raions_with_data]

        # Filter to raions with data
        if len(active_raion_keys) < n_raions:
            keep_indices = raions_with_data
            features = features[:, keep_indices, :]
            mask = mask[:, keep_indices]
            active_raions = active_raion_keys

        print(f"  Active raions: {len(active_raions)}")
        print(f"  DeepState: {mask.sum()} raion-day observations, {self.N_FEATURES} features")

        return features, mask, active_raions, dates


# =============================================================================
# EXPANDED FIRMS RAION LOADER
# =============================================================================

# Import FIRMS data paths
FIRMS_DIR = DATA_DIR / "firms"
FIRMS_ARCHIVE_FILE = FIRMS_DIR / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"
FIRMS_NRT_FILE = FIRMS_DIR / "DL_FIRE_SV-C2_706038" / "fire_nrt_SV-C2_706038.csv"


class FIRMSExpandedRaionLoader:
    """
    Expanded FIRMS raion loader with 35 decomposed features.

    NASA FIRMS provides fire hotspot detections with rich metadata including
    confidence levels, radiative power, timing, and satellite source.

    Output: Per-raion daily feature tensor [n_days, n_raions, 35 features]

    Features (35 total):
    - [0-2]: Count metrics (total_fires, high_confidence, low_confidence)
    - [3-6]: Confidence breakdown (nominal, low, high, ultra_high)
    - [7-10]: FRP statistics (frp_sum, frp_mean, frp_max, frp_std)
    - [11-14]: Brightness stats (brightness_mean, brightness_max, brightness_min, brightness_std)
    - [15-18]: Timing (day_fires, night_fires, dawn_fires, dusk_fires)
    - [19-22]: Temporal patterns (morning_peak, afternoon_peak, evening_peak, night_peak)
    - [23-26]: Satellite source (viirs_snpp, viirs_noaa20, modis, other_sat)
    - [27-30]: Intensity categories (low_intensity, medium_intensity, high_intensity, extreme_intensity)
    - [31-34]: Derived (fire_density, frp_per_fire, day_night_ratio, fire_clustering)
    """

    FEATURE_NAMES = [
        # Count metrics [0-2]
        'total_fires', 'high_confidence_fires', 'low_confidence_fires',
        # Confidence breakdown [3-6]
        'confidence_nominal', 'confidence_low', 'confidence_high', 'confidence_ultra',
        # FRP statistics [7-10]
        'frp_sum', 'frp_mean', 'frp_max', 'frp_std',
        # Brightness stats [11-14]
        'brightness_mean', 'brightness_max', 'brightness_min', 'brightness_std',
        # Timing [15-18]
        'day_fires', 'night_fires', 'dawn_fires', 'dusk_fires',
        # Temporal patterns [19-22]
        'morning_peak', 'afternoon_peak', 'evening_peak', 'night_peak',
        # Satellite source [23-26]
        'viirs_snpp', 'viirs_noaa20', 'modis_detections', 'other_satellite',
        # Intensity categories [27-30]
        'low_intensity', 'medium_intensity', 'high_intensity', 'extreme_intensity',
        # Derived [31-34]
        'fire_density', 'frp_per_fire', 'day_night_ratio', 'fire_clustering',
    ]

    FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
    N_FEATURES = len(FEATURE_NAMES)

    def __init__(
        self,
        raion_manager: Optional[RaionBoundaryManager] = None,
        archive_file: Optional[Path] = None,
        nrt_file: Optional[Path] = None,
    ):
        """Initialize FIRMS expanded loader."""
        self.raion_manager = raion_manager or RaionBoundaryManager()
        self.archive_file = archive_file or FIRMS_ARCHIVE_FILE
        self.nrt_file = nrt_file or FIRMS_NRT_FILE
        self._data: Optional[pd.DataFrame] = None

    def load_raw(self) -> pd.DataFrame:
        """Load combined FIRMS archive and NRT data."""
        if self._data is not None:
            return self._data

        dfs = []

        if self.archive_file.exists():
            print(f"Loading FIRMS archive from {self.archive_file}...")
            archive_df = pd.read_csv(self.archive_file)
            archive_df['source_file'] = 'archive'
            dfs.append(archive_df)
            print(f"  Archive: {len(archive_df)} hotspots")

        if self.nrt_file.exists():
            print(f"Loading FIRMS NRT from {self.nrt_file}...")
            nrt_df = pd.read_csv(self.nrt_file)
            nrt_df['source_file'] = 'nrt'
            dfs.append(nrt_df)
            print(f"  NRT: {len(nrt_df)} hotspots")

        if not dfs:
            warnings.warn("No FIRMS data files found")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
        df = df.dropna(subset=['acq_date', 'latitude', 'longitude'])

        # Filter to Ukraine bounds
        ukraine_mask = (
            (df['latitude'] >= 44.0) & (df['latitude'] <= 53.0) &
            (df['longitude'] >= 22.0) & (df['longitude'] <= 41.0)
        )
        df = df[ukraine_mask].copy()

        print(f"  Total hotspots in Ukraine: {len(df)}")
        self._data = df
        return df

    def load_raion_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        active_raions: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[datetime]]:
        """
        Load FIRMS hotspots as per-raion daily features.

        Extracts 35 decomposed features for comprehensive fire analysis.
        """
        start = start_date or DEFAULT_START_DATE
        end = end_date or DEFAULT_END_DATE

        df = self.load_raw()
        if df.empty:
            return np.array([]), np.array([]), [], []

        df = df[(df['acq_date'] >= start) & (df['acq_date'] <= end)].copy()

        self.raion_manager.load()

        # Assign raions
        print("  Assigning FIRMS hotspots to raions...")
        raion_assignments = []
        for _, row in df.iterrows():
            raion = self.raion_manager.get_raion_for_point(row['latitude'], row['longitude'])
            raion_assignments.append(raion)
        df['raion'] = raion_assignments

        # Get active raions
        if active_raions is None:
            raion_counts = df['raion'].value_counts()
            active_raions = raion_counts[raion_counts >= 10].index.tolist()
            active_raions = [r for r in active_raions if r is not None]

        print(f"  Active raions: {len(active_raions)}")

        # Create date range
        n_days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(n_days)]
        date_to_idx = {d.date(): i for i, d in enumerate(dates)}

        raion_to_idx = {r: i for i, r in enumerate(active_raions)}
        n_raions = len(active_raions)

        # Initialize arrays
        features = np.zeros((n_days, n_raions, self.N_FEATURES), dtype=np.float32)
        mask = np.zeros((n_days, n_raions), dtype=bool)

        # Temporary storage for aggregation
        day_raion_fires: Dict[Tuple[int, int], List[Dict]] = {}

        for _, row in df.iterrows():
            raion = row.get('raion')
            if raion is None or raion not in raion_to_idx:
                continue

            date_key = row['acq_date'].date()
            if date_key not in date_to_idx:
                continue

            day_idx = date_to_idx[date_key]
            raion_idx = raion_to_idx[raion]
            key = (day_idx, raion_idx)

            if key not in day_raion_fires:
                day_raion_fires[key] = []

            # Parse acquisition time to hour
            acq_time = row.get('acq_time', 1200)
            try:
                hour = int(acq_time) // 100 if acq_time else 12
            except (ValueError, TypeError):
                hour = 12

            day_raion_fires[key].append({
                'brightness': row.get('brightness', 0) or 0,
                'frp': row.get('frp', 0) or 0,
                'confidence': str(row.get('confidence', '')).lower(),
                'daynight': row.get('daynight', 'D'),
                'satellite': str(row.get('satellite', '')).lower(),
                'hour': hour,
            })

        # Aggregate features
        for (day_idx, raion_idx), fires in day_raion_fires.items():
            mask[day_idx, raion_idx] = True
            feat = features[day_idx, raion_idx]

            n_fires = len(fires)
            brightnesses = [f['brightness'] for f in fires if f['brightness'] > 0]
            frps = [f['frp'] for f in fires if f['frp'] > 0]
            confidences = [f['confidence'] for f in fires]
            daynights = [f['daynight'] for f in fires]
            satellites = [f['satellite'] for f in fires]
            hours = [f['hour'] for f in fires]

            # Count metrics [0-2]
            feat[self.FEATURE_IDX['total_fires']] = n_fires
            feat[self.FEATURE_IDX['high_confidence_fires']] = sum(
                1 for c in confidences if c in ['h', 'high', '100']
            )
            feat[self.FEATURE_IDX['low_confidence_fires']] = sum(
                1 for c in confidences if c in ['l', 'low', '0']
            )

            # Confidence breakdown [3-6]
            feat[self.FEATURE_IDX['confidence_nominal']] = sum(1 for c in confidences if c in ['n', 'nominal', '50'])
            feat[self.FEATURE_IDX['confidence_low']] = sum(1 for c in confidences if c in ['l', 'low', '0', '25'])
            feat[self.FEATURE_IDX['confidence_high']] = sum(1 for c in confidences if c in ['h', 'high', '75', '100'])
            feat[self.FEATURE_IDX['confidence_ultra']] = sum(1 for c in confidences if c in ['100'])

            # FRP statistics [7-10]
            if frps:
                feat[self.FEATURE_IDX['frp_sum']] = sum(frps)
                feat[self.FEATURE_IDX['frp_mean']] = np.mean(frps)
                feat[self.FEATURE_IDX['frp_max']] = max(frps)
                feat[self.FEATURE_IDX['frp_std']] = np.std(frps) if len(frps) > 1 else 0

            # Brightness stats [11-14]
            if brightnesses:
                feat[self.FEATURE_IDX['brightness_mean']] = np.mean(brightnesses)
                feat[self.FEATURE_IDX['brightness_max']] = max(brightnesses)
                feat[self.FEATURE_IDX['brightness_min']] = min(brightnesses)
                feat[self.FEATURE_IDX['brightness_std']] = np.std(brightnesses) if len(brightnesses) > 1 else 0

            # Timing [15-18]
            feat[self.FEATURE_IDX['day_fires']] = sum(1 for d in daynights if d == 'D')
            feat[self.FEATURE_IDX['night_fires']] = sum(1 for d in daynights if d == 'N')
            feat[self.FEATURE_IDX['dawn_fires']] = sum(1 for h in hours if 4 <= h < 8)
            feat[self.FEATURE_IDX['dusk_fires']] = sum(1 for h in hours if 18 <= h < 22)

            # Temporal patterns [19-22]
            feat[self.FEATURE_IDX['morning_peak']] = sum(1 for h in hours if 6 <= h < 12)
            feat[self.FEATURE_IDX['afternoon_peak']] = sum(1 for h in hours if 12 <= h < 18)
            feat[self.FEATURE_IDX['evening_peak']] = sum(1 for h in hours if 18 <= h < 24)
            feat[self.FEATURE_IDX['night_peak']] = sum(1 for h in hours if 0 <= h < 6)

            # Satellite source [23-26]
            feat[self.FEATURE_IDX['viirs_snpp']] = sum(1 for s in satellites if 'snpp' in s or 'npp' in s)
            feat[self.FEATURE_IDX['viirs_noaa20']] = sum(1 for s in satellites if 'noaa20' in s or 'j01' in s)
            feat[self.FEATURE_IDX['modis_detections']] = sum(1 for s in satellites if 'modis' in s or 'terra' in s or 'aqua' in s)
            feat[self.FEATURE_IDX['other_satellite']] = n_fires - (
                feat[self.FEATURE_IDX['viirs_snpp']] +
                feat[self.FEATURE_IDX['viirs_noaa20']] +
                feat[self.FEATURE_IDX['modis_detections']]
            )

            # Intensity categories [27-30] based on FRP
            if frps:
                feat[self.FEATURE_IDX['low_intensity']] = sum(1 for f in frps if f < 10)
                feat[self.FEATURE_IDX['medium_intensity']] = sum(1 for f in frps if 10 <= f < 50)
                feat[self.FEATURE_IDX['high_intensity']] = sum(1 for f in frps if 50 <= f < 200)
                feat[self.FEATURE_IDX['extreme_intensity']] = sum(1 for f in frps if f >= 200)

            # Derived features [31-34]
            if n_fires > 0:
                feat[self.FEATURE_IDX['frp_per_fire']] = (
                    feat[self.FEATURE_IDX['frp_sum']] / n_fires if feat[self.FEATURE_IDX['frp_sum']] > 0 else 0
                )
                day_count = feat[self.FEATURE_IDX['day_fires']]
                night_count = feat[self.FEATURE_IDX['night_fires']]
                if night_count > 0:
                    feat[self.FEATURE_IDX['day_night_ratio']] = day_count / night_count
                else:
                    feat[self.FEATURE_IDX['day_night_ratio']] = day_count

        # Compute fire clustering [34]
        for raion_idx in range(n_raions):
            for day_idx in range(n_days):
                window_start = max(0, day_idx - 3)
                window_end = min(n_days, day_idx + 4)
                cluster_count = sum(
                    features[d, raion_idx, self.FEATURE_IDX['total_fires']]
                    for d in range(window_start, window_end)
                    if d != day_idx
                )
                features[day_idx, raion_idx, self.FEATURE_IDX['fire_clustering']] = cluster_count

        print(f"  FIRMS: {mask.sum()} raion-day observations, {self.N_FEATURES} features")

        return features, mask, active_raions, dates


# =============================================================================
# COMBINED LOADER
# =============================================================================

class CombinedRaionLoader:
    """
    Combined loader for all raion-level data sources with expanded features.

    Provides a unified interface to load and align features from:
    - Geoconfirmed (50 features): Equipment losses with side attribution, status, attack methods
    - Air Raid Sirens (30 features): Alert patterns with duration, intensity, temporal
    - UCDP (35 features): Conflict events with casualties, actor attribution, certainty
    - Warspotting (33 features): Verified equipment losses with status, tags
    - DeepState (48 features): Military units, frontlines, territorial control
    - FIRMS (35 features): Fire hotspots with confidence, FRP, timing

    Total potential features: 231 per raion per day (aligned across sources)
    """

    # Combined feature names with source prefixes to avoid collisions
    # Order: geoconfirmed (50) + air_raid_sirens (30) + ucdp (35) + warspotting (33)
    #        + deepstate (48) + firms_expanded (35) = 231 total
    FEATURE_NAMES = (
        [f"geoconfirmed_{name}" for name in GeoconfirmedRaionLoader.FEATURE_NAMES] +
        [f"air_raid_sirens_{name}" for name in AirRaidSirensRaionLoader.FEATURE_NAMES] +
        [f"ucdp_{name}" for name in UCDPRaionLoader.FEATURE_NAMES] +
        [f"warspotting_{name}" for name in WarspottingRaionLoader.FEATURE_NAMES] +
        [f"deepstate_{name}" for name in DeepStateRaionLoader.FEATURE_NAMES] +
        [f"firms_expanded_{name}" for name in FIRMSExpandedRaionLoader.FEATURE_NAMES]
    )

    FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
    N_FEATURES = len(FEATURE_NAMES)  # Should be 231

    def __init__(
        self,
        raion_manager: Optional[RaionBoundaryManager] = None,
    ):
        """Initialize combined loader with shared raion manager."""
        self.raion_manager = raion_manager or RaionBoundaryManager()

        self.geoconfirmed = GeoconfirmedRaionLoader(self.raion_manager)
        self.air_raid = AirRaidSirensRaionLoader(self.raion_manager)
        self.ucdp = UCDPRaionLoader(self.raion_manager)
        self.warspotting = WarspottingRaionLoader(self.raion_manager)
        self.deepstate = DeepStateRaionLoader(self.raion_manager)
        self.firms = FIRMSExpandedRaionLoader(self.raion_manager)

    def load_all_sources(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_raion_observations: int = 20,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str], List[datetime]]]:
        """
        Load all sources with aligned raions.

        Args:
            start_date: Start of date range
            end_date: End of date range
            min_raion_observations: Minimum total observations for a raion

        Returns:
            Dict mapping source name to (features, mask, raion_keys, dates)
        """
        start = start_date or DEFAULT_START_DATE
        end = end_date or DEFAULT_END_DATE

        # Load raion manager
        self.raion_manager.load()

        # First pass: determine active raions across all sources
        print("Determining active raions across all sources...")

        # Load raw data to count raion observations
        gc_df = self.geoconfirmed.load_raw()
        ar_df = self.air_raid.load_raw()
        ucdp_df = self.ucdp.load_raw()

        raion_obs_counts: Dict[str, int] = {}

        # Count Geoconfirmed
        for _, row in gc_df.iterrows():
            if pd.isna(row['lat']) or pd.isna(row['lon']):
                continue
            raion = self.raion_manager.get_raion_for_point(row['lat'], row['lon'])
            if raion:
                raion_obs_counts[raion] = raion_obs_counts.get(raion, 0) + 1

        # Count UCDP
        for _, row in ucdp_df.iterrows():
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                continue
            raion = self.raion_manager.get_raion_for_point(row['latitude'], row['longitude'])
            if raion:
                raion_obs_counts[raion] = raion_obs_counts.get(raion, 0) + 1

        # Get active raions
        active_raions = [r for r, c in raion_obs_counts.items() if c >= min_raion_observations]
        active_raions = sorted(active_raions)
        print(f"Active raions (>={min_raion_observations} observations): {len(active_raions)}")

        # Load each source with common raion list
        results = {}

        print("\nLoading Geoconfirmed...")
        results['geoconfirmed'] = self.geoconfirmed.load_raion_features(
            start, end, active_raions
        )

        print("\nLoading Air Raid Sirens...")
        results['air_raid'] = self.air_raid.load_raion_features(
            start, end, active_raions
        )

        print("\nLoading UCDP...")
        results['ucdp'] = self.ucdp.load_raion_features(
            start, end, active_raions
        )

        print("\nLoading Warspotting...")
        results['warspotting'] = self.warspotting.load_raion_features(
            start, end, active_raions
        )

        print("\nLoading DeepState...")
        results['deepstate'] = self.deepstate.load_raion_features(
            start, end, active_raions
        )

        print("\nLoading FIRMS...")
        results['firms'] = self.firms.load_raion_features(
            start, end, active_raions
        )

        # Summary of feature counts
        print("\n" + "=" * 60)
        print("Feature Decomposition Summary")
        print("=" * 60)
        feature_counts = {
            'geoconfirmed': GeoconfirmedRaionLoader.N_FEATURES,
            'air_raid': AirRaidSirensRaionLoader.N_FEATURES,
            'ucdp': UCDPRaionLoader.N_FEATURES,
            'warspotting': WarspottingRaionLoader.N_FEATURES,
            'deepstate': DeepStateRaionLoader.N_FEATURES,
            'firms': FIRMSExpandedRaionLoader.N_FEATURES,
        }
        total_features = sum(feature_counts.values())
        for source, count in feature_counts.items():
            print(f"  {source}: {count} features")
        print(f"  TOTAL: {total_features} features per raion per day")

        return results


# =============================================================================
# TESTING
# =============================================================================

def test_loaders():
    """Test all new raion loaders."""
    print("=" * 60)
    print("Testing New Source Raion Loaders")
    print("=" * 60)

    combined = CombinedRaionLoader()

    results = combined.load_all_sources(
        start_date=datetime(2022, 2, 24),
        end_date=datetime(2024, 12, 31),
        min_raion_observations=10,
    )

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    for source, (features, mask, raions, dates) in results.items():
        if len(features) > 0:
            print(f"\n{source}:")
            print(f"  Shape: {features.shape}")
            print(f"  Raions: {len(raions)}")
            print(f"  Days: {len(dates)}")
            print(f"  Observations: {mask.sum()}")
            print(f"  Coverage: {100 * mask.sum() / mask.size:.1f}%")
        else:
            print(f"\n{source}: No data loaded")


if __name__ == "__main__":
    test_loaders()
