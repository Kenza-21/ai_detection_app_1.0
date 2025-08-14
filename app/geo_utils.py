import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import logging
import time
from math import radians, sin, cos, sqrt, atan2

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache local pour éviter de géocoder plusieurs fois les mêmes adresses
GEO_CACHE = {}

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcule la distance en kilomètres entre deux points géographiques
    en utilisant la formule de Haversine.
    """
    R = 6371.0  # Rayon de la Terre en km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def geocode_location(country, city, retries=3):
    """
    Convertit une localisation (pays, ville) en coordonnées géographiques.
    Utilise un cache pour éviter les requêtes répétées.
    """
    if pd.isna(country) or pd.isna(city) or not str(country).strip() or not str(city).strip():
        return None, None

    cache_key = f"{country.lower()}_{city.lower()}"
    
    if cache_key in GEO_CACHE:
        return GEO_CACHE[cache_key]

    geolocator = Nominatim(user_agent="fraud_detection_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    location_str = f"{city}, {country}"
    location = None

    for attempt in range(retries):
        try:
            location = geocode(location_str)
            if location:
                break
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt == retries - 1:
                logger.warning(f"Échec du géocodage pour {location_str}: {str(e)}")
                return None, None
            time.sleep(1)

    if location:
        GEO_CACHE[cache_key] = (location.latitude, location.longitude)
        return location.latitude, location.longitude

    return None, None

def add_geo_features(df):
    """
    Ajoute les caractéristiques géographiques au DataFrame:
    - Coordonnées (latitude, longitude) pour débiteur et créancier
    - Distance entre les deux parties
    """
    try:
        # Géocodage des adresses débiteur
        df[['debtor_lat', 'debtor_lon']] = df.apply(
            lambda x: geocode_location(x['debtor_country'], x['debtor_city']), 
            axis=1, result_type='expand'
        )

        # Géocodage des adresses créancier
        df[['creditor_lat', 'creditor_lon']] = df.apply(
            lambda x: geocode_location(x['creditor_country'], x['creditor_city']), 
            axis=1, result_type='expand'
        )

        # Calcul de la distance
        df['distance_km'] = df.apply(
            lambda x: haversine_distance(
                x['debtor_lat'], x['debtor_lon'],
                x['creditor_lat'], x['creditor_lon']
            ) if pd.notna(x['debtor_lat']) and pd.notna(x['creditor_lat']) else np.nan,
            axis=1
        )

        return df
    except Exception as e:
        logger.error(f"Erreur dans add_geo_features: {str(e)}")
        raise