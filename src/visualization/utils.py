"""Generic functions for visualizations."""
import glob
import os
import re
import time

from PIL import Image
from selenium import webdriver
from src.settings import CHROME_DRIVER_PATH, CHROME_PATH, TMP_REPORTS_DIRECOTRY


def _atoi(text: str):
    return int(text) if text.isdigit() else text


def _natural_keys(text):
    return [_atoi(c) for c in re.split(r'(\d+)', text)]


def pngs_to_gif(in_path: str, out_path: str):
    """Combine png files into gif.

    Args:
        in_path (str): files to combine (ex hour_*.png)
        out_path (str): resulting gif path
    """
    img, *imgs = [
        Image.open(f) for f in sorted(glob.glob(in_path), key=_natural_keys)
    ]
    img.save(
        fp=out_path, format='GIF', append_images=imgs,
        save_all=True, duration=250, loop=0
    )


def plotly_to_png(fig, path):
    html_file = os.path.join(TMP_REPORTS_DIRECOTRY, 'plot.html')

    fig.write_html(html_file)

    options = webdriver.ChromeOptions()

    #  TODO - fix this to be generic
    options.binary_location = CHROME_PATH
    chrome_driver_binary = CHROME_DRIVER_PATH

    driver = webdriver.Chrome(chrome_driver_binary, options=options)
    driver.get(html_file)
    print('x')
    time.sleep(1)
    driver.save_screenshot(path)
