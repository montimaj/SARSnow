from scipy.ndimage import gaussian_filter
from osgeo import gdal
from collections import defaultdict
import glob
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np

def read_c2_matrices(dir, pattern = '*.tif'):
    image_dict = defaultdict(lambda: [])
    dir += os.sep + pattern
    for files in glob.glob(dir):
        date = files[files.find('_') + 1: files.rfind('_')]
        date = datetime.datetime.strptime(date, '%d%b%y')
        image_dict[date].append(gdal.Open(files))
    return image_dict

image_dict = read_c2_matrices(r'C:\Users\s6038174\Downloads\ITC\SAYANTAN\shashi kumar\pmat\TIFs')
bands = []
for v in image_dict.values():
    for images in v:
        print(images.RasterCount)
        bands.append(images.GetRasterBand(1).ReadAsArray())

cpd = np.rad2deg(bands[0])
cpd = gaussian_filter(cpd, sigma=3)
plt.imshow(cpd)
plt.colorbar()
plt.show()
cpd = cpd[~np.isnan(cpd)]
print(np.mean(cpd))