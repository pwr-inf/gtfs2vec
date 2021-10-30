"""Code from https://github.com/uber/h3-py-notebooks"""

import json

import folium
import geojson
import h3


def visualize_hexagons(hexagons, color="red", folium_map=None):
    """
    hexagons is a list of hexcluster. Each hexcluster is a list of hexagons.
    eg. [[hex1, hex2], [hex3, hex4]]
    """
    polylines = []
    lat = []
    lng = []
    for hex in hexagons:
        polygons = h3.h3_set_to_multi_polygon([hex], geo_json=False)
        # flatten polygons into loops.
        outlines = [loop for polygon in polygons for loop in polygon]
        polyline = [outline + [outline[0]] for outline in outlines][0]
        lat.extend(map(lambda v: v[0], polyline))
        lng.extend(map(lambda v: v[1], polyline))
        polylines.append(polyline)

    if folium_map is None:
        m = folium.Map(
            location=[sum(lat) / len(lat), sum(lng) / len(lng)],
            zoom_start=11,
            tiles='cartodbpositron'
        )
    else:
        m = folium_map
    for polyline in polylines:
        my_PolyLine = folium.PolyLine(locations=polyline, weight=3, color=color)
        m.add_child(my_PolyLine)
    return m


def visualize_polygon(polyline, color):
    polyline.append(polyline[0])
    lat = [p[0] for p in polyline]
    lng = [p[1] for p in polyline]
    m = folium.Map(
        location=[sum(lat) / len(lat), sum(lng) / len(lng)],
        zoom_start=13,
        tiles='cartodbpositron'
    )
    my_PolyLine = folium.PolyLine(locations=polyline, weight=8, color=color)
    m.add_child(my_PolyLine)
    return m


def hexagons_dataframe_to_geojson(df_hex, hex_id_field,
                                  geometry_field, value_field):

    """Produce the GeoJSON representation containing all geometries in a dataframe
     based on a column in geojson format (geometry_field)"""

    list_features = []

    for i, row in df_hex.iterrows():
        feature = geojson.Feature(geometry=row[geometry_field],
                                  id=row[hex_id_field],
                                  properties={"value": row[value_field]})
        list_features.append(feature)

    feat_collection = geojson.FeatureCollection(list_features)

    geojson_result = json.dumps(feat_collection)

    return geojson_result
