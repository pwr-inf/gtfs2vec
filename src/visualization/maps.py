"""Map related visualization tools."""
import json
import os
import time
from pathlib import Path
from typing import List, Union

import folium
import pandas as pd
from selenium import webdriver
from src.configurations import MAPS_SETUPS
from src.data.h3 import get_hex_from_city_geojson
from src.settings import (CHROME_DRIVER_PATH, CHROME_PATH,
                          CITIES_POLYGONS_DIRECTORY, TMP_REPORTS_DIRECOTRY)
from src.visualization.uber_h3 import hexagons_dataframe_to_geojson


def create_choropleth(
        geo_data: str,
        df_data: pd.DataFrame,
        value_column: str,
        df_id_col: str = 'hex_id',
        geo_id_col: str = 'id',
        initial_position: List[float] = [51.1239095, 17.0055833],
        initial_zoom: float = 11,
        label: str = "",
        threshold_scale: List[int] = None,
        fill_color: str = 'YlOrRd'
) -> folium.Map:
    """Creates choropleth map using data from df and geojson.

    Args:
        geo_data (str): Geojson formatted data
        df_data (pd.DataFrame): df with additional data
        value_column (str): value from df_data to show on map
        df_id_col (str, optional): name of column in df wich matches geojson.
        Defaults to 'hex_id'.
        geo_id_col (str, optional): name of field in geojson wich matches df.
        Defaults to 'id'.
        initial_position (List[float], optional): initial map centering.
        Defaults to [51.1239095, 17.0055833] - Wroclaw.
        initial_zoom (float, optional): initial map zoom.
        Defaults to 11 - Wroclaw.
        area_name (str, optional): name to append to map title.
        Defaults to "".

    Returns:
        folium.Map: generated choropleth map.

    """
    m = folium.Map(
        location=initial_position,
        zoom_start=initial_zoom
    )

    choropleth = folium.Choropleth(
        geo_data=geo_data,
        data=df_data,
        columns=[df_id_col, value_column],
        key_on=geo_id_col,
        threshold_scale=threshold_scale,
        fill_color=fill_color,
        fill_opacity=0.55,
        line_opacity=0.2,
        legend_name=f'{label}',
        highlight=True
    ).add_to(m)

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['value'], labels=False)
    )

    return m


def map_to_png(m: folium.Map, path: str):
    """Saving map to png using selenium.

    Args:
        m (folium.Map): folium map
        path (str): path where to save png
    """
    html_file = os.path.join(TMP_REPORTS_DIRECOTRY, 'map.html')

    m.save(html_file)

    options = webdriver.ChromeOptions()

    #  TODO - fix this to be generic
    options.binary_location = CHROME_PATH
    chrome_driver_binary = CHROME_DRIVER_PATH

    driver = webdriver.Chrome(chrome_driver_binary, options=options)
    driver.set_window_size(1000, 1000)
    driver.get(html_file)
    time.sleep(1)
    driver.save_screenshot(path)


def plot_clusters(city: str, clusters_df: pd.DataFrame, n_clusters: int, save_html_path: Union[str, None] = None, label: str = "") -> folium.Map:
    """Plot clusters on choropleth map.

    Args:
        city (str): city name matching files
        clusters_df (pd.DataFrame): df with hex_id and cluster
        n_clusters (int): number of clusters in data
        save_html_path (Union[str, None], optional): path to save html. Defaults to None.
        label (str). label for map. Defaults to "".

    Returns:
        folium.Map: choropleth map
    """
    with open(os.path.join(CITIES_POLYGONS_DIRECTORY, f'{city.lower()}.geojson')) as f:
        city_geojson = json.load(f)

    city_geojson = city_geojson['features'][0]

    city_df, _ = get_hex_from_city_geojson(city_geojson, resolution=8)

    city_df = city_df.merge(
        clusters_df['cluster'].reset_index(), on='hex_id', how='left')

    city_geo_data = hexagons_dataframe_to_geojson(
        city_df, 'hex_id', 'geometry_dict', 'cluster')

    if city not in MAPS_SETUPS.keys() or MAPS_SETUPS[city] is None:
        m = create_choropleth(
            city_geo_data,
            city_df,
            value_column='cluster',
            label=label,
            fill_color='Set1',
            threshold_scale=list(range(n_clusters + 1))
        )
    else:
        m = create_choropleth(
            city_geo_data,
            city_df,
            value_column='cluster',
            label=label,
            fill_color='Set1',
            threshold_scale=list(range(n_clusters + 1)),
            initial_position=MAPS_SETUPS[city]['position'],
            initial_zoom=MAPS_SETUPS[city]['zoom']
        )

    if save_html_path is not None:
        Path(save_html_path).mkdir(parents=True, exist_ok=True)
        m.save(os.path.join(save_html_path, f'{city}.html'))

    return m
