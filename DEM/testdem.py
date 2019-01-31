from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

dem = gdal.Open('Doris_Cards/dem.tiff')
dem_arr = dem.GetRasterBand(1).ReadAsArray()
#print(dem_arr)
#np.savetxt('demtxt.txt', dem_arr)
plt.hist(dem_arr)
#print(dem_arr[dem_arr == 0])
#plt.imshow(dem_arr)
plt.show()