"""Utils related to gtfs feed."""
import json
import os
from typing import List, Union

import gtfs_kit as gk
import pandas as pd
from src.configurations import CITIES_GEOJSON, CITIES_GTFS, CITIES_SIGSPATIAL
from src.data.h3 import get_hex_from_city_geojson
from src.settings import (CITIES_POLYGONS_DIRECTORY, GTFS_CACHE_DIRECTORY,
                          GTFS_DATA_DIRECTORY)
from tqdm.auto import tqdm

import h3


def count_trips_from_stop(
    feed: gk.Feed,
    date: str,
    time_resolution: str = '1H'
) -> pd.DataFrame:
    """Calculates sum of trips at given resolution in one day.

    Args:
        feed (gk.Feed): gtfs feed
        date (str): date to consider
        time_resolution (str, optional): Defaults to '1H'.

    Returns:
        pd.DataFrame: num_trips for each hour at each stop_id
    """
    ts = feed.compute_stop_time_series([date], freq=time_resolution)

    records = []

    for idx, row in ts.iterrows():
        h = idx.hour
        for s, n in row['num_trips'].items():
            records.append((s, h, n))

    df = pd.DataFrame(records, columns=['stop_id', 'hour', 'num_trips'])

    return df


def count_trips_from_hex(
    feed: gk.Feed,
    date: str,
    time_resolution: str = "1H",
    hex_resolution: int = 8
) -> pd.DataFrame:
    """Calculates sum of trips aggregated to hex.

    Args:
        feed (gk.Feed): gfts feed
        date (str): date to consider
        time_resolution (str, optional): Defaults to "1H".
        hex_resolution (int, optional): Defaults to 8.

    Returns:
        pd.DataFrame: num trips at each hour for all hexagons.
    """
    tqdm.pandas()
    stops_num_trips = count_trips_from_stop(feed, date, time_resolution)
    stops_num_trips = stops_num_trips.merge(feed.stops, on='stop_id')
    stops_num_trips['hex_id'] = stops_num_trips.apply(
        lambda row: h3.geo_to_h3(row['stop_lat'], row['stop_lon'], hex_resolution), axis=1)
    sum_by_hex = stops_num_trips.groupby(['hour', 'hex_id'])['num_trips'].sum()

    df = sum_by_hex.reset_index().pivot_table(
        values='num_trips', index='hex_id', columns='hour', aggfunc='first')
    df = df.add_prefix('trips_at_')

    return df


def count_destiantions_from_hex(
    feed: gk.Feed,
    hex_resolution: int = 8,
    include_sum: bool = True
) -> pd.DataFrame:
    """Calculates available directions aggregated to hex.

    Args:
        feed (gk.Feed): gtfs feed
        hex_resolution (int, optional): Defaults to 8.

    Returns:
        pd.DataFrame: num of directions at each hour + whole day sum for all hexagons.
    """
    tqdm.pandas()
    df = feed.stop_times.merge(feed.trips, on='trip_id')
    df = df.merge(feed.stops, on='stop_id')

    df = df[df['departure_time'].notna()]

    df['hour'] = df['departure_time'].apply(
        lambda x: int(x[:2].replace(':', '')) % 24)
    df['hex_id'] = df.apply(lambda row: h3.geo_to_h3(
        row['stop_lat'], row['stop_lon'], hex_resolution), axis=1)

    directions_per_hour = df.groupby(['hex_id', 'hour']).agg({
        'trip_headsign': 'nunique'})
    pivoted = directions_per_hour.pivot_table(
        values='trip_headsign', index='hex_id', columns='hour', aggfunc='first', fill_value=0)
    pivoted = pivoted.add_prefix('directions_at_')

    if include_sum:
        directions_whole_day = df.loc[(df['hour'] >= 6) & (df['hour'] < 23)].groupby(['hex_id']).agg({'trip_headsign': 'nunique'}).rename(
            columns={'trip_headsign': 'directions_whole_day'})
        pivoted = pivoted.merge(directions_whole_day, on='hex_id')

    return pivoted


def load_data_for_city(
    city: str,
    limit: bool,
    add_city_name: bool = False,
    geojson_file: Union[str, None] = None
) -> pd.DataFrame:
    """Loads data for a single city and calculates features.

    Args:
        city (str): city name matching gtfs file name
        limit (bool): whether to limit hexes to city boundaries
        city_name (bool): add column with city name to resulting df. Defaults to False.
        geojson_file (Union[str, None]): geojson file name if different from gtfs file. Defaults to None.

    Returns:
        pd.DataFrame: features for hex in a city
    """

    gtfs_path = os.path.join(GTFS_DATA_DIRECTORY, f'{city}.zip')
    feed = gk.read_feed(gtfs_path, dist_units='km')
    if 'Barcelona-tram' in city or 'Lipsk' in city:
        date = feed.get_week(2)[2]
    elif 'Ryga' in city:
        date = feed.get_week(260)[2]
    elif 'Wilno' in city:
        date = feed.get_week(80)[2]
    else:
        date = feed.get_week(1)[2]

    trips = count_trips_from_hex(feed, date)
    destinations = count_destiantions_from_hex(feed)

    df = trips.merge(destinations, on='hex_id')

    if limit:
        if geojson_file is None:
            with open(os.path.join(CITIES_POLYGONS_DIRECTORY, f'{city.lower()}.geojson')) as f:
                city_geojson = json.load(f)
        else:
            with open(os.path.join(CITIES_POLYGONS_DIRECTORY, f'{geojson_file.lower()}.geojson')) as f:
                city_geojson = json.load(f)

        city_geojson = city_geojson['features'][0]
        city_df, _ = get_hex_from_city_geojson(city_geojson, resolution=8)

        df = df.merge(city_df, on='hex_id', how='inner').drop(
            columns=['geometry', 'geometry_dict'])

        df = df.set_index('hex_id')

    if add_city_name:
        df['city'] = city

    return df


def load_data_from_multiple_gtfs(
        files: List[str],
        limit: bool,
        geojson_file: str,
        add_city_name: bool = False) -> pd.DataFrame:
    """Loads data for a city which has multiple gtfs files.

    Args:
        files (List[str]): file names list (without .zip extension)
        limit (bool): whether to limit hexes to city boundaries
        geojson_file (str): geojson file for city boundaries.
        city_name (bool): add column with city name to resulting df. Defaults to False.

    Returns:
        pd.DataFrame: features for hex in a city
    """
    df = pd.DataFrame()
    for city in files:
        part_df = load_data_for_city(
            city=city,
            geojson_file=geojson_file,
            limit=limit,
            add_city_name=False)

        df = df.add(part_df, fill_value=0)

    if add_city_name:
        df['city'] = geojson_file

    return df


def _load_from_cache(city: str) -> pd.DataFrame:
    df = pd.read_pickle(os.path.join(GTFS_CACHE_DIRECTORY, f'{city}.pkl.gz'))
    return df


def _store_cache(city: str, df: pd.DataFrame):
    df.to_pickle(os.path.join(GTFS_CACHE_DIRECTORY, f'{city}.pkl.gz'))


def load_data(limit: bool = True, add_city_name: bool = True, cache: bool = True, cities: Union[List[str], None] = None) -> pd.DataFrame:
    """Load data using config in `src/configurations.py`.

    Args:
        limit (bool, optional): whether to limit hexes to city boundaries. Defaults to True.
        add_city_name (bool, optional): add column with city name to resulting df. Defaults to True.
        cache (bool, optional): wheter to use cache to save/load data. Defaults to True.
        cities (Union[List[str], None], optional): cities list if different in configurations.py

    Returns:
        pd.DataFrame: features for hex in all cities
    """
    cities_data = []

    if cities is None:
        cities = CITIES_SIGSPATIAL

    for city in tqdm(cities):
        if cache and os.path.exists(os.path.join(GTFS_CACHE_DIRECTORY, f'{city}.pkl.gz')):
            cities_data.append(_load_from_cache(city))
        else:
            city_df = load_data_from_multiple_gtfs(
                files=CITIES_GTFS[city], geojson_file=CITIES_GEOJSON[city],
                limit=limit, add_city_name=add_city_name
            )
            cities_data.append(city_df)

            if cache:
                _store_cache(city, city_df)

    df = pd.concat(cities_data).fillna(0)

    return df
