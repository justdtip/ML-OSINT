"""
COMPREHENSIVE FEATURE HIERARCHY FOR UKRAINE CONFLICT OSINT DATA

This module defines the complete feature hierarchy across all data sources,
enabling granular neural network analysis with decomposed features.

Structure:
    SOURCE -> FEATURE -> SUBFEATURE -> SUB-SUBFEATURE

Each leaf node can be used as an input to the neural network.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto


# =============================================================================
# ENUMS FOR CATEGORICAL DECOMPOSITIONS
# =============================================================================

class ViolenceType(Enum):
    """UCDP violence type categories"""
    STATE_BASED = 1      # Government vs. armed group
    NON_STATE = 2        # Armed group vs. armed group
    ONE_SIDED = 3        # Violence against civilians


class UkraineOblast(Enum):
    """Ukrainian oblasts for regional decomposition"""
    DONETSK = "Donetsk oblast"
    LUHANSK = "Luhansk oblast"
    KHARKIV = "Kharkiv oblast"
    KHERSON = "Kherson oblast"
    ZAPORIZHZHYA = "Zaporizhzhya oblast"
    SUMY = "Sumy oblast"
    DNIPROPETROVSK = "Dnipropetrovsk oblast"
    MYKOLAYIV = "Mykolayiv oblast"
    CHERNIHIV = "Chernihiv oblast"
    KYIV_OBLAST = "Kyiv oblast"
    KYIV_CITY = "Kyiv Special Republican city"
    ODESSA = "Odessa oblast"
    ZHYTOMYR = "Zhytomyr oblast"
    CRIMEA = "Autonomous Republic of Crimea"
    POLTAVA = "Poltava oblast"
    OTHER = "Other"


class ConflictParty(Enum):
    """Parties to the conflict"""
    GOV_UKRAINE = "Government of Ukraine"
    GOV_RUSSIA = "Government of Russia (Soviet Union)"
    DPR = "DPR"
    LPR = "LPR"
    CIVILIANS = "Civilians"
    OTHER = "Other"


class FireConfidence(Enum):
    """FIRMS fire detection confidence"""
    HIGH = "h"
    NOMINAL = "n"
    LOW = "l"


class FireType(Enum):
    """FIRMS fire type classification"""
    PRESUMED_VEGETATION = 0
    ACTIVE_FIRE = 2
    OTHER_STATIC = 3


class ArrowDirection(Enum):
    """DeepState attack direction arrows (16 compass directions)"""
    N = 1
    NNE = 2
    NE = 3
    ENE = 4
    E = 5
    ESE = 6
    SE = 7
    SSE = 8
    S = 9
    SSW = 10
    SW = 11
    WSW = 12
    W = 13
    WNW = 14
    NW = 15
    NNW = 16


class UnitEchelon(Enum):
    """Military unit echelon levels"""
    ARMY = "army"
    DIVISION = "division"
    BRIGADE = "brigade"
    REGIMENT = "regiment"
    BATTALION = "battalion"
    SQUAD = "squad"


class TerritoryStatus(Enum):
    """DeepState territory status"""
    OCCUPIED = "occupied"
    LIBERATED = "liberated"
    CONTESTED = "contested"
    UNKNOWN = "unknown"


class TankSeries(Enum):
    """Tank series classification"""
    T62 = "T-62"
    T64 = "T-64"
    T72 = "T-72"
    T80 = "T-80"
    T90 = "T-90"
    OTHER = "Other"


class AFVSeries(Enum):
    """Armored Fighting Vehicle series"""
    BMP = "BMP"
    BTR = "BTR"
    MTLB = "MT-LB"
    BMD = "BMD"
    OTHER = "Other"


# =============================================================================
# FEATURE HIERARCHY DEFINITION
# =============================================================================

FEATURE_HIERARCHY = {
    # =========================================================================
    # 1. UCDP (Uppsala Conflict Data Program)
    # =========================================================================
    "ucdp": {
        "name": "UCDP Conflict Events",
        "source_file": "data/ucdp/ged_events.csv",
        "total_columns": 47,
        "features": {
            # --- Aggregate Features ---
            "events": {
                "description": "Total conflict events",
                "type": "numeric",
                "subfeatures": {
                    "by_violence_type": {
                        "state_based": "Government vs armed group (type=1)",
                        "non_state": "Armed group vs armed group (type=2)",
                        "one_sided": "Violence against civilians (type=3)"
                    },
                    "by_clarity": {
                        "clear": "High certainty events (clarity=1)",
                        "uncertain": "Lower certainty events (clarity=2)"
                    },
                    "by_precision": {
                        "exact_location": "where_prec=1 (exact coordinates)",
                        "approximate": "where_prec=2-4 (settlement/district level)",
                        "regional": "where_prec=5-7 (oblast/country level)"
                    }
                }
            },
            "deaths": {
                "description": "Fatalities from conflict",
                "type": "numeric",
                "subfeatures": {
                    "by_party": {
                        "deaths_side_a": "Ukrainian government forces deaths",
                        "deaths_side_b": "Russian/separatist deaths",
                        "deaths_civilians": "Civilian casualties",
                        "deaths_unknown": "Unattributed deaths"
                    },
                    "by_estimate": {
                        "best_estimate": "Most likely death count",
                        "high_estimate": "Upper bound estimate",
                        "low_estimate": "Lower bound estimate"
                    }
                }
            },
            "geography": {
                "description": "Geographic decomposition",
                "type": "categorical",
                "subfeatures": {
                    "by_oblast": {
                        "donetsk": "Donetsk oblast events/deaths",
                        "luhansk": "Luhansk oblast events/deaths",
                        "kharkiv": "Kharkiv oblast events/deaths",
                        "kherson": "Kherson oblast events/deaths",
                        "zaporizhzhya": "Zaporizhzhya oblast events/deaths",
                        "sumy": "Sumy oblast events/deaths",
                        "dnipropetrovsk": "Dnipropetrovsk oblast",
                        "mykolayiv": "Mykolayiv oblast",
                        "chernihiv": "Chernihiv oblast",
                        "kyiv_oblast": "Kyiv oblast",
                        "kyiv_city": "Kyiv city",
                        "odessa": "Odessa oblast",
                        "other": "All other oblasts"
                    },
                    "by_front": {
                        "eastern": "Donetsk + Luhansk (main front)",
                        "southern": "Zaporizhzhya + Kherson",
                        "northeastern": "Kharkiv + Sumy",
                        "northern": "Kyiv + Chernihiv",
                        "rear": "All other areas"
                    }
                }
            }
        }
    },

    # =========================================================================
    # 2. FIRMS (Fire Information for Resource Management System)
    # =========================================================================
    "firms": {
        "name": "FIRMS Fire Detections",
        "source_file": "data/firms/DL_FIRE_SV-C2_706038/fire_archive_SV-C2_706038.csv",
        "total_columns": 15,
        "features": {
            "fire_count": {
                "description": "Number of fire detections",
                "type": "numeric",
                "subfeatures": {
                    "by_daynight": {
                        "day_fires": "Daytime detections (D)",
                        "night_fires": "Nighttime detections (N)"
                    },
                    "by_confidence": {
                        "high_confidence": "High confidence detections (h)",
                        "nominal_confidence": "Nominal confidence (n)",
                        "low_confidence": "Low confidence (l)"
                    },
                    "by_type": {
                        "type_0": "Presumed vegetation fire",
                        "type_2": "Active fire detection",
                        "type_3": "Other static source"
                    }
                }
            },
            "frp": {
                "description": "Fire Radiative Power (intensity)",
                "type": "numeric",
                "subfeatures": {
                    "by_intensity": {
                        "frp_tiny": "0-5 MW (small burns)",
                        "frp_small": "5-20 MW (moderate fires)",
                        "frp_medium": "20-50 MW (significant fires)",
                        "frp_large": "50-100 MW (large fires)",
                        "frp_very_large": "100-500 MW (very large)",
                        "frp_extreme": ">500 MW (extreme, likely industrial/military)"
                    },
                    "by_daynight": {
                        "frp_day_mean": "Mean FRP during daytime",
                        "frp_night_mean": "Mean FRP during nighttime",
                        "frp_day_max": "Max FRP during daytime",
                        "frp_night_max": "Max FRP during nighttime"
                    },
                    "by_time": {
                        "frp_morning": "06:00-12:00 local",
                        "frp_afternoon": "12:00-18:00 local",
                        "frp_evening": "18:00-00:00 local",
                        "frp_night": "00:00-06:00 local"
                    }
                }
            },
            "brightness": {
                "description": "Fire brightness temperature",
                "type": "numeric",
                "subfeatures": {
                    "brightness_t21": "21µm channel temperature",
                    "brightness_t31": "31µm channel temperature",
                    "brightness_ratio": "T21/T31 ratio (fire characterization)"
                }
            },
            "scan_track": {
                "description": "Pixel dimensions (fire size proxy)",
                "type": "numeric",
                "subfeatures": {
                    "scan": "Along-scan pixel dimension",
                    "track": "Along-track pixel dimension",
                    "pixel_area": "scan * track (fire area proxy)"
                }
            }
        }
    },

    # =========================================================================
    # 3. SENTINEL SATELLITE DATA
    # =========================================================================
    "sentinel": {
        "name": "Sentinel Satellite Imagery",
        "source_files": [
            "data/sentinel/sentinel_timeseries_raw.json",
            "data/sentinel/sentinel_categories_overview.json"
        ],
        "features": {
            "sentinel_1": {
                "description": "Sentinel-1 SAR Radar",
                "type": "count",
                "subfeatures": {
                    "by_polarization": {
                        "vv": "Vertical-Vertical polarization",
                        "vh": "Vertical-Horizontal polarization",
                        "vv_vh_ratio": "VV/VH ratio (surface classification)"
                    },
                    "by_product": {
                        "grd": "Ground Range Detected",
                        "slc": "Single Look Complex"
                    },
                    "derived_indices": {
                        "rvi": "Radar Vegetation Index",
                        "change_detection": "Coherent change detection score"
                    }
                }
            },
            "sentinel_2": {
                "description": "Sentinel-2 Optical Multispectral",
                "type": "count",
                "subfeatures": {
                    "by_processing_level": {
                        "l1c": "Top-of-Atmosphere reflectance",
                        "l2a": "Surface reflectance (atmospherically corrected)"
                    },
                    "by_band": {
                        "b01_coastal": "Coastal aerosol (60m)",
                        "b02_blue": "Blue (10m)",
                        "b03_green": "Green (10m)",
                        "b04_red": "Red (10m)",
                        "b05_veg_red_edge": "Vegetation red edge 1 (20m)",
                        "b06_veg_red_edge": "Vegetation red edge 2 (20m)",
                        "b07_veg_red_edge": "Vegetation red edge 3 (20m)",
                        "b08_nir": "NIR (10m)",
                        "b8a_veg_red_edge": "Vegetation red edge 4 (20m)",
                        "b09_water_vapor": "Water vapor (60m)",
                        "b11_swir": "SWIR 1 (20m)",
                        "b12_swir": "SWIR 2 (20m)"
                    },
                    "cloud_metrics": {
                        "cloud_cover_pct": "Cloud cover percentage",
                        "cloud_free_count": "Cloud-free scene count",
                        "avg_cloud": "Average cloud cover"
                    },
                    "derived_indices": {
                        "ndvi": "(B08-B04)/(B08+B04) - Vegetation",
                        "ndwi": "(B03-B08)/(B03+B08) - Water",
                        "nbr": "(B08-B12)/(B08+B12) - Burn ratio",
                        "ndbi": "(B11-B08)/(B11+B08) - Built-up"
                    }
                }
            },
            "sentinel_3": {
                "description": "Sentinel-3 Fire/Land/Ocean",
                "type": "count",
                "subfeatures": {
                    "by_product": {
                        "slstr_frp": "Fire Radiative Power (1km)",
                        "olci_lfr": "Land Full Resolution (300m)",
                        "olci_wfr": "Water Full Resolution"
                    },
                    "derived": {
                        "otci": "OLCI Terrestrial Chlorophyll Index",
                        "gifapar": "Global Instantaneous FAPAR"
                    }
                }
            },
            "sentinel_5p": {
                "description": "Sentinel-5P Atmospheric",
                "type": "count",
                "subfeatures": {
                    "trace_gases": {
                        "no2": "Nitrogen Dioxide column",
                        "co": "Carbon Monoxide column",
                        "so2": "Sulfur Dioxide column",
                        "ch4": "Methane column",
                        "o3": "Ozone column"
                    },
                    "aerosols": {
                        "aai": "Aerosol Absorbing Index",
                        "aod": "Aerosol Optical Depth"
                    }
                }
            }
        }
    },

    # =========================================================================
    # 4. DEEPSTATE FRONT LINE DATA
    # =========================================================================
    "deepstate": {
        "name": "DeepState Front Line Map",
        "source_files": [
            "data/deepstate/snapshots/deepstate_full_*.json",
            "data/deepstate/wayback_snapshots/*.json"
        ],
        "features": {
            "polygons": {
                "description": "Territorial control polygons (113 total)",
                "type": "count/area",
                "subfeatures": {
                    "by_status": {
                        "occupied": "Russian-controlled territory (red, #a52714)",
                        "liberated": "Ukrainian-controlled (green, #0f9d58)",
                        "contested": "Gray zone / active combat (gray, #bcaaa4)",
                        "unknown": "Status unclear"
                    },
                    "by_region": {
                        "crimea": "Crimean peninsula",
                        "ordlo": "ORDLO (DPR/LPR 2014 lines)",
                        "other_occupied": "Occupied since Feb 2022"
                    },
                    "metrics": {
                        "polygon_count": "Number of polygons per status",
                        "polygon_area": "Total area per status (sq km)",
                        "boundary_length": "Front line length (km)"
                    }
                }
            },
            "attack_directions": {
                "description": "Attack direction arrows (82 total)",
                "type": "count",
                "subfeatures": {
                    "by_cardinal": {
                        "north_attacks": "arrows 15,16,1,2 (N, NNW, NNE, NE)",
                        "east_attacks": "arrows 3,4,5,6 (ENE, E, ESE, SE)",
                        "south_attacks": "arrows 7,8,9,10 (SSE, S, SSW, SW)",
                        "west_attacks": "arrows 11,12,13,14 (WSW, W, WNW, NW)"
                    },
                    "by_specific": {
                        f"arrow_{i}": f"Direction {i}" for i in range(1, 17)
                    },
                    "metrics": {
                        "total_arrows": "Total attack direction count",
                        "dominant_direction": "Most common attack direction"
                    }
                }
            },
            "military_units": {
                "description": "Unit markers (256 + 15 total)",
                "type": "count",
                "subfeatures": {
                    "by_echelon": {
                        "armies": "Army-level HQs (icon-4)",
                        "divisions": "Division-level units",
                        "brigades": "Brigade-level units",
                        "regiments": "Regiment-level units",
                        "battalions": "Battalion-level units"
                    },
                    "by_type": {
                        "motorized_rifle": "Motorized rifle units",
                        "tank": "Tank units",
                        "artillery": "Artillery units",
                        "airborne": "VDV/airborne units",
                        "reconnaissance": "Recon units",
                        "bars": "BARS volunteer units",
                        "akhmat": "Akhmat (Chechen) units"
                    },
                    "metrics": {
                        "unit_count": "Total unit markers",
                        "unit_density": "Units per km of front"
                    }
                }
            },
            "airfields": {
                "description": "Airfield markers (54 total)",
                "type": "count",
                "subfeatures": {
                    "by_region": {
                        "crimea_airfields": "Crimean peninsula",
                        "eastern_airfields": "Eastern Ukraine",
                        "northern_airfields": "Belarus/northern",
                        "western_airfields": "Western Ukraine"
                    },
                    "by_status": {
                        "operational": "Active military airfields",
                        "damaged": "Confirmed damage",
                        "unknown_status": "Unknown operational status"
                    }
                }
            },
            "special_markers": {
                "description": "Special event markers (2)",
                "type": "categorical",
                "subfeatures": {
                    "moskva_cruiser": "Moskva sinking location",
                    "crimean_bridge": "Kerch bridge status"
                }
            }
        }
    },

    # =========================================================================
    # 5. EQUIPMENT LOSSES
    # =========================================================================
    "equipment": {
        "name": "Equipment Losses (UKR MOD + Oryx)",
        "source_files": [
            "data/war-losses-data/2022-Ukraine-Russia-War-Dataset/data/russia_losses_equipment.json",
            "data/war-losses-data/2022-Ukraine-Russia-War-Dataset/data/russia_losses_equipment_oryx.json"
        ],
        "features": {
            "aircraft": {
                "description": "Fixed-wing aircraft losses",
                "type": "cumulative",
                "subfeatures": {
                    "by_type": {
                        "sukhoi_fighters": "Su-24, Su-25, Su-30, Su-34, Su-35",
                        "mig_fighters": "MiG-29, MiG-31",
                        "transport": "Il-76, An-26",
                        "awacs": "A-50"
                    }
                }
            },
            "helicopters": {
                "description": "Helicopter losses",
                "type": "cumulative",
                "subfeatures": {
                    "by_type": {
                        "ka52": "Ka-52 Alligator attack helicopter",
                        "mi28": "Mi-28 Havoc attack helicopter",
                        "mi24_35": "Mi-24/35 Hind attack helicopter",
                        "mi8": "Mi-8 transport helicopter",
                        "other_heli": "Other helicopters"
                    }
                }
            },
            "tanks": {
                "description": "Main battle tank losses",
                "type": "cumulative",
                "subfeatures": {
                    "by_series": {
                        "t62_series": "T-62, T-62M, T-62MV",
                        "t64_series": "T-64A, T-64BV",
                        "t72_series": "T-72A/B/B3 (most common)",
                        "t80_series": "T-80BV, T-80U, T-80BVM",
                        "t90_series": "T-90A, T-90M"
                    },
                    "by_generation": {
                        "soviet_era": "T-62, T-64, early T-72/T-80",
                        "modern": "T-72B3, T-80BVM, T-90M"
                    }
                }
            },
            "apc_ifv": {
                "description": "APCs and IFVs",
                "type": "cumulative",
                "subfeatures": {
                    "by_series": {
                        "bmp_series": "BMP-1, BMP-2, BMP-3",
                        "btr_series": "BTR-80, BTR-82A",
                        "mtlb_series": "MT-LB family",
                        "bmd_series": "BMD airborne",
                        "other_afv": "Other AFVs"
                    }
                }
            },
            "artillery": {
                "description": "Artillery systems",
                "type": "cumulative",
                "subfeatures": {
                    "by_type": {
                        "field_artillery": "Towed guns",
                        "self_propelled": "2S19, 2S1, 2S3, etc.",
                        "mrl": "BM-21, BM-27, BM-30, TOS-1"
                    }
                }
            },
            "drones": {
                "description": "UAV losses",
                "type": "cumulative",
                "subfeatures": {
                    "by_role": {
                        "recon_uav": "Orlan-10, Orlan-30, etc.",
                        "combat_uav": "Lancet, Shahed-136",
                        "loitering": "Loitering munitions"
                    }
                }
            },
            "air_defense": {
                "description": "Air defense systems",
                "type": "cumulative",
                "subfeatures": {
                    "by_type": {
                        "sam_long_range": "S-300, S-400",
                        "sam_medium": "Buk",
                        "sam_short": "Tor, Pantsir",
                        "spaa": "ZSU-23, Tunguska"
                    }
                }
            },
            "naval": {
                "description": "Naval vessel losses",
                "type": "cumulative",
                "subfeatures": {
                    "by_class": {
                        "cruiser": "Moskva",
                        "landing_ship": "Ropucha, Saratov",
                        "patrol_boat": "Raptor, patrol boats",
                        "other_naval": "Support vessels"
                    }
                }
            },
            "vehicles": {
                "description": "Military vehicles",
                "type": "cumulative",
                "subfeatures": {
                    "trucks": "Ural, KamAZ trucks",
                    "fuel_tanks": "Fuel tankers",
                    "command_posts": "Mobile command posts",
                    "engineering": "Engineering vehicles"
                }
            }
        }
    },

    # =========================================================================
    # 6. PERSONNEL LOSSES
    # =========================================================================
    "personnel": {
        "name": "Personnel Losses (UKR MOD estimates)",
        "source_file": "data/war-losses-data/2022-Ukraine-Russia-War-Dataset/data/russia_losses_personnel.json",
        "features": {
            "cumulative": {
                "description": "Total cumulative casualties",
                "type": "cumulative"
            },
            "daily_change": {
                "description": "Daily casualty rate",
                "type": "delta"
            },
            "monthly_rate": {
                "description": "Monthly casualty aggregation",
                "type": "aggregate"
            }
        }
    },

    # =========================================================================
    # 7. VIINA TERRITORIAL CONTROL (NEW)
    # =========================================================================
    "viina": {
        "name": "VIINA Territorial Control",
        "source_file": "data/viina/extracted/control_latest_*.csv",
        "total_columns": 8,
        "features": {
            "control_status": {
                "description": "Territorial control by locality",
                "type": "categorical/count",
                "subfeatures": {
                    "by_status": {
                        "localities_ua_control": "Localities under Ukrainian control",
                        "localities_ru_control": "Localities under Russian control",
                        "localities_contested": "Contested/gray zone localities",
                        "localities_unknown": "Unknown status localities"
                    },
                    "by_percentage": {
                        "pct_ua_control": "Percentage under Ukrainian control",
                        "pct_ru_control": "Percentage under Russian control",
                        "pct_contested": "Percentage contested"
                    }
                }
            },
            "control_changes": {
                "description": "Daily territorial changes",
                "type": "delta",
                "subfeatures": {
                    "ua_changes": {
                        "localities_gained_ua": "Localities gained by Ukraine",
                        "localities_lost_ua": "Localities lost by Ukraine"
                    },
                    "ru_changes": {
                        "localities_gained_ru": "Localities gained by Russia",
                        "localities_lost_ru": "Localities lost by Russia"
                    }
                }
            },
            "source_agreement": {
                "description": "Agreement between VIINA data sources",
                "type": "quality",
                "subfeatures": {
                    "sources_agree_pct": "Overall source agreement percentage",
                    "wiki_dsm_agree": "Wikipedia-DeepState agreement",
                    "wiki_isw_agree": "Wikipedia-ISW agreement",
                    "dsm_isw_agree": "DeepState-ISW agreement"
                }
            },
            "derived_metrics": {
                "description": "Computed territorial metrics",
                "type": "derived",
                "subfeatures": {
                    "front_activity_index": "Total localities changing hands daily",
                    "control_momentum": "7-day trend direction",
                    "control_volatility": "7-day change volatility"
                }
            }
        }
    },

    # =========================================================================
    # 8. HDX CONFLICT EVENTS (NEW)
    # =========================================================================
    "hdx_conflict": {
        "name": "HDX Conflict Events",
        "source_file": "data/hdx/ukraine/conflict_events_2022_present.csv",
        "features": {
            "events": {
                "description": "Conflict event counts",
                "type": "count",
                "subfeatures": {
                    "by_type": {
                        "events_civilian_targeting": "Civilian targeting events",
                        "events_battles": "Battle events",
                        "events_explosions": "Explosions/remote violence",
                        "events_protests": "Protest events",
                        "events_other": "Other event types"
                    },
                    "by_region": {
                        "events_donetsk": "Donetsk oblast events",
                        "events_kharkiv": "Kharkiv oblast events",
                        "events_kherson": "Kherson oblast events",
                        "events_zaporizhzhia": "Zaporizhzhia oblast events",
                        "events_other_regions": "Events in other regions"
                    }
                }
            },
            "fatalities": {
                "description": "Fatality metrics",
                "type": "count",
                "subfeatures": {
                    "fatalities_total": "Total fatalities",
                    "fatalities_per_event": "Average fatalities per event",
                    "fatalities_max_event": "Maximum single-event fatalities"
                }
            },
            "intensity": {
                "description": "Conflict intensity metrics",
                "type": "derived",
                "subfeatures": {
                    "intensity_index": "Events × fatality rate",
                    "regional_spread": "Number of affected regions"
                }
            }
        }
    },

    # =========================================================================
    # 9. HDX FOOD PRICES (NEW)
    # =========================================================================
    "hdx_food": {
        "name": "HDX Food Prices",
        "source_file": "data/hdx/ukraine/food_prices_2022_present.csv",
        "features": {
            "price_statistics": {
                "description": "Price distribution metrics",
                "type": "numeric",
                "subfeatures": {
                    "avg_price": "Average commodity price",
                    "median_price": "Median commodity price",
                    "price_std": "Price standard deviation",
                    "price_range": "Price range (max - min)"
                }
            },
            "category_prices": {
                "description": "Prices by commodity category",
                "type": "numeric",
                "subfeatures": {
                    "cereals_avg": "Cereals/tubers average price",
                    "vegetables_avg": "Vegetables/fruits average price",
                    "meat_avg": "Meat/fish/eggs average price",
                    "dairy_avg": "Dairy average price",
                    "oils_avg": "Oils/fats average price"
                }
            },
            "price_dynamics": {
                "description": "Price change metrics",
                "type": "delta",
                "subfeatures": {
                    "price_change_pct": "Period-over-period change %",
                    "price_7day_trend": "7-period price trend",
                    "price_volatility": "Price volatility"
                }
            },
            "food_security": {
                "description": "Food security indicators",
                "type": "derived",
                "subfeatures": {
                    "inflation_proxy": "Inflation indicator",
                    "food_security_index": "Food security index",
                    "price_anomaly_score": "Price anomaly score"
                }
            }
        }
    },

    # =========================================================================
    # 10. HDX RAINFALL (NEW)
    # =========================================================================
    "hdx_rainfall": {
        "name": "HDX Rainfall Data",
        "source_file": "data/hdx/ukraine/rainfall_2022_present.csv",
        "features": {
            "rainfall_statistics": {
                "description": "Rainfall measurements",
                "type": "numeric",
                "subfeatures": {
                    "rainfall_mean": "Mean rainfall",
                    "rainfall_median": "Median rainfall",
                    "rainfall_max": "Maximum rainfall",
                    "rainfall_std": "Rainfall standard deviation"
                }
            },
            "anomaly_metrics": {
                "description": "Rainfall anomaly analysis",
                "type": "numeric",
                "subfeatures": {
                    "anomaly_pct_mean": "Mean rainfall anomaly %",
                    "above_normal_pct": "% of areas above normal",
                    "below_normal_pct": "% of areas below normal"
                }
            },
            "risk_indices": {
                "description": "Agricultural risk indicators",
                "type": "derived",
                "subfeatures": {
                    "drought_risk_index": "Drought risk score",
                    "flood_risk_index": "Flood risk score",
                    "rainfall_vs_lta_ratio": "Rainfall to long-term average ratio"
                }
            }
        }
    },

    # =========================================================================
    # 11. IOM DISPLACEMENT (NEW)
    # =========================================================================
    "iom_displacement": {
        "name": "IOM Displacement Data",
        "source_file": "data/iom/ukr-iom-dtm-from-api-admin-0-to-1.csv",
        "features": {
            "total_idps": {
                "description": "Total IDP counts",
                "type": "count",
                "subfeatures": {
                    "total_idps": "Total internally displaced persons",
                    "idps_male": "Male IDPs",
                    "idps_female": "Female IDPs"
                }
            },
            "destination_regions": {
                "description": "IDP destinations",
                "type": "count",
                "subfeatures": {
                    "idps_kyiv": "IDPs in Kyiv region",
                    "idps_lviv": "IDPs in Lviv region",
                    "idps_dnipro": "IDPs in Dnipro region",
                    "idps_kharkiv": "IDPs in Kharkiv region",
                    "idps_zaporizhzhia": "IDPs in Zaporizhzhia region",
                    "idps_other": "IDPs in other regions"
                }
            },
            "origin_regions": {
                "description": "IDP origins",
                "type": "count",
                "subfeatures": {
                    "from_donetsk": "IDPs from Donetsk",
                    "from_luhansk": "IDPs from Luhansk",
                    "from_other_conflict": "IDPs from other conflict areas"
                }
            },
            "derived_metrics": {
                "description": "Displacement indicators",
                "type": "derived",
                "subfeatures": {
                    "displacement_intensity": "Normalized displacement level",
                    "gender_ratio": "Female to male IDP ratio",
                    "avg_per_region": "Average IDPs per receiving region"
                }
            }
        }
    },

    # =========================================================================
    # 12. GOOGLE MOBILITY (NEW - Limited timeframe)
    # =========================================================================
    "google_mobility": {
        "name": "Google Mobility (Feb-Oct 2022)",
        "source_file": "data/wayback_archives/google_mobility/ukraine_only/*.csv",
        "date_range": "2022-02-24 to 2022-10-15",
        "features": {
            "mobility_categories": {
                "description": "Mobility by location type",
                "type": "percent_change",
                "subfeatures": {
                    "retail_recreation": "Retail & recreation mobility change %",
                    "grocery_pharmacy": "Grocery & pharmacy mobility change %",
                    "parks": "Parks mobility change %",
                    "transit_stations": "Transit stations mobility change %",
                    "workplaces": "Workplace mobility change %",
                    "residential": "Residential mobility change %"
                }
            },
            "regional_mobility": {
                "description": "Regional average mobility",
                "type": "percent_change",
                "subfeatures": {
                    "kyiv_avg": "Kyiv average mobility",
                    "lviv_avg": "Lviv average mobility",
                    "dnipro_avg": "Dnipro average mobility",
                    "kharkiv_avg": "Kharkiv average mobility",
                    "odesa_avg": "Odesa average mobility"
                }
            },
            "derived_indices": {
                "description": "Computed mobility indicators",
                "type": "derived",
                "subfeatures": {
                    "economic_activity_index": "(retail + workplaces + transit) / 3",
                    "social_distancing_index": "Residential vs other mobility",
                    "war_impact_index": "Mobility decline indicator",
                    "recovery_index": "Mobility recovery from baseline",
                    "regional_disparity": "Regional mobility variance"
                }
            }
        }
    }
}


# =============================================================================
# FEATURE COUNT SUMMARY
# =============================================================================

def count_features(hierarchy: dict, level: int = 0) -> tuple:
    """
    Recursively count features at each level of the hierarchy.
    Returns (sources, features, subfeatures, sub_subfeatures)
    """
    sources = 0
    features = 0
    subfeatures = 0
    sub_subfeatures = 0

    for key, value in hierarchy.items():
        if level == 0:
            sources += 1
            if 'features' in value:
                f_count = len(value['features'])
                features += f_count
                for feat_name, feat_data in value['features'].items():
                    if 'subfeatures' in feat_data:
                        sf_count = len(feat_data['subfeatures'])
                        subfeatures += sf_count
                        for sf_name, sf_data in feat_data['subfeatures'].items():
                            if isinstance(sf_data, dict):
                                sub_subfeatures += len(sf_data)

    return sources, features, subfeatures, sub_subfeatures


def get_all_leaf_features() -> List[str]:
    """
    Extract all leaf-level feature names for use as neural network inputs.
    Returns a flat list of feature paths like 'ucdp.deaths.by_party.deaths_side_a'
    """
    leaves = []

    def traverse(obj, path=""):
        if isinstance(obj, dict):
            if 'features' in obj:
                traverse(obj['features'], path)
            elif 'subfeatures' in obj:
                traverse(obj['subfeatures'], path)
            else:
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(value, dict) and 'subfeatures' in value:
                        traverse(value['subfeatures'], new_path)
                    elif isinstance(value, dict):
                        traverse(value, new_path)
                    else:
                        leaves.append(new_path)

    for source, data in FEATURE_HIERARCHY.items():
        if 'features' in data:
            for feat_name, feat_data in data['features'].items():
                feat_path = f"{source}.{feat_name}"
                if 'subfeatures' in feat_data:
                    for sf_name, sf_data in feat_data['subfeatures'].items():
                        sf_path = f"{feat_path}.{sf_name}"
                        if isinstance(sf_data, dict):
                            for ssf_name in sf_data.keys():
                                leaves.append(f"{sf_path}.{ssf_name}")
                        else:
                            leaves.append(sf_path)
                else:
                    leaves.append(feat_path)

    return leaves


# =============================================================================
# MAIN - PRINT SUMMARY
# =============================================================================

if __name__ == "__main__":
    sources, features, subfeatures, sub_subfeatures = count_features(FEATURE_HIERARCHY)

    print("=" * 80)
    print("FEATURE HIERARCHY SUMMARY")
    print("=" * 80)
    print(f"\nData Sources: {sources}")
    print(f"Top-Level Features: {features}")
    print(f"Subfeatures: {subfeatures}")
    print(f"Sub-Subfeatures (leaf nodes): {sub_subfeatures}")

    leaf_features = get_all_leaf_features()
    print(f"\nTotal leaf features for ANN input: {len(leaf_features)}")

    print("\n" + "=" * 80)
    print("LEAF FEATURE PATHS (first 50):")
    print("=" * 80)
    for i, feat in enumerate(leaf_features[:50]):
        print(f"  {i+1:3}. {feat}")

    if len(leaf_features) > 50:
        print(f"  ... and {len(leaf_features) - 50} more")

    print("\n" + "=" * 80)
    print("SOURCE BREAKDOWN:")
    print("=" * 80)

    for source, data in FEATURE_HIERARCHY.items():
        print(f"\n{source.upper()}: {data['name']}")
        if 'features' in data:
            for feat_name, feat_data in data['features'].items():
                desc = feat_data.get('description', '')
                sf_count = len(feat_data.get('subfeatures', {}))
                print(f"  - {feat_name}: {desc}")
                if sf_count > 0:
                    print(f"    ({sf_count} subfeatures)")
