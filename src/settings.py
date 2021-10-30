"""General settings for paths etc."""
import os

PROJECT_DIRECTORY = os.path.normpath(
    os.path.join(os.path.dirname(__file__), ".."))
DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "datasets")
REPORTS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'reports')
MODELS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'models')
EMBEDDINGS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'embeddings')
MAPS_REPORTS_DIRECTORY = os.path.join(REPORTS_DIRECTORY, 'maps')
PLOTS_REPORTS_DIRECTORY = os.path.join(REPORTS_DIRECTORY, 'plots')
TMP_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "tmp")
TMP_REPORTS_DIRECOTRY = os.path.join(REPORTS_DIRECTORY, 'tmp')
TIMETABLES_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'timetables')
GTFS_DATA_DIRECTORY = os.path.join(TIMETABLES_DATA_DIRECTORY, "gtfs")
GTFS_CACHE_DIRECTORY = os.path.join(TIMETABLES_DATA_DIRECTORY, 'cache')
CITIES_POLYGONS_DIRECTORY = os.path.join(DATA_DIRECTORY, 'cities')

if os.name == 'nt':
    # FIXME: add path for selenium which is used to save maps
    CHROME_PATH = "Path to chrome.exe"
    CHROME_DRIVER_PATH = "Path to chromedriver.exe"
else:
    pass  # FIXME: add path to chrome and driver for linux
