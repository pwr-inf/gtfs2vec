"""Utils for h3 related operations."""
from copy import deepcopy
from typing import List, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

import h3


def get_hex_from_city_geojson(
    city: dict,
    resolution: int = 8
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """Fills city poolygon with hexagons at a given resolution.

    Args:
        city (dict): Geojson formatted city polygon
        resolution (int, optional): Uber H3 resolution. Defaults to 8.

    Returns:
        Tuple[gpd.GeoDataFrame, List[str]]: geodataframe with hexagons
        in city and list of hexagons ids.

    """
    city = deepcopy(city)
    city['geometry']['coordinates'][0] = [
        [x[1], x[0]] for x in city['geometry']['coordinates'][0]
    ]
    hexagons = h3.polyfill(city['geometry'], resolution)

    df = pd.DataFrame(hexagons, columns=['hex_id'])

    df['geometry_dict'] = df['hex_id'].apply(
        lambda x: {
            "type": "Polygon",
            "coordinates": [h3.h3_to_geo_boundary(h=x, geo_json=True)]
        })

    df['geometry'] = df['hex_id'].apply(
        lambda x: Polygon(h3.h3_to_geo_boundary(h=x, geo_json=True)))

    gdf = gpd.GeoDataFrame(df)

    return gdf, hexagons


def get_stops_from_stops_file(
    stops_txt: str,
    resolution: int = 8
) -> pd.DataFrame:
    """Read stops info from stops.txt file and match with hexagons.

    Args:
        stops_txt (str): stops file

    Returns:
        pd.DataFrame: df with stops and their hexagons

    """
    df = pd.read_csv(stops_txt)

    df['hex_id'] = df.apply(
        lambda row:
        h3.geo_to_h3(row['stop_lat'], row['stop_lon'], resolution), axis=1)

    return df
