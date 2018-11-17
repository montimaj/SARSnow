from osgeo import gdal
import numpy as np
import affine
import cv2
from ellipse2geoid import EGM96
import utm

NO_DATA_VALUE = -32768

def csv_merge(csvfiles):
    data = ""
    for csv in csvfiles:
        csv = open(csv, 'r')
        data += csv.read()
        csv.close()
    merged = open('Merged.csv', 'w')
    merged.write(data)
    merged.close()


def retrieve_pixel_coords(geo_coord, data_source):
    x, y = geo_coord[0], geo_coord[1]
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    return px, py


def retrieve_geo_coords(pixel_coords, data_source):
    x, y = pixel_coords[0], pixel_coords[1]
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    gx, gy = forward_transform * (x, y)
    return gx, gy


def write_dem_tif(dem_arr, src_file, outfile='test'):
    driver = gdal.GetDriverByName("GTiff")
    absdem = driver.Create(outfile + '.tif', dem_arr.shape[1], dem_arr.shape[0], 1, gdal.GDT_Float32)
    absdem.SetProjection(src_file.GetProjection())
    absdem.SetGeoTransform(src_file.GetGeoTransform())
    absdem.GetRasterBand(1).SetNoDataValue(NO_DATA_VALUE)
    absdem.GetRasterBand(1).WriteArray(dem_arr)
    absdem.FlushCache()


def rel2absdem(rdem, gcps):
    gcps = open(gcps, 'r')
    ground_points = gcps.readlines()
    rdem_arr = rdem.GetRasterBand(1).ReadAsArray()
    #rdem_arr[rdem_arr == NO_DATA_VALUE] = np.nan
    #kernel = np.ones((32, 32), np.float32)/32**2
    #rdem_arr = cv2.filter2D(rdem_arr, -1, kernel)
    #rdem_arr = cv2.bilateralFilter(rdem_arr, 50, 200, 200)
    zero_abs_height = []
    ground_pixels = []
    for gcp in ground_points:
        val = gcp.split(',')
        x = round(float(val[1]))
        y = round(float(val[2]))
        #latlon = utm.to_latlon(x, y, 43, 'U')
        latlon = x, y
        z = float(val[3])
        px, py = retrieve_pixel_coords(latlon, rdem)
        print(px, py)
        if px <= rdem_arr.shape[0] and py <= rdem_arr.shape[1]:
            rel_height = rdem_arr[py, px]
            if rel_height != NO_DATA_VALUE:
                if rel_height < 0:
                    zero_abs_height.append(z - rel_height)
                elif rel_height > 0:
                    zero_abs_height.append(rel_height - z)
                else:
                    zero_abs_height.append(z)
                ground_pixels.append((px, py, z))
    zero_abs_height = np.mean(zero_abs_height)
    print(zero_abs_height)
    rdem_arr += zero_abs_height
    for pixels in ground_pixels:
        rdem_arr[pixels[1], pixels[0]] = pixels[2]
    #rdem_arr[np.isnan(rdem_arr)] = NO_DATA_VALUE
    print('Min=', np.min(rdem_arr[rdem_arr > 0]), 'Max=', np.max(rdem_arr[rdem_arr > 0]))
    return rdem_arr


def ellipse2ortho(dem_arr, src_dem):
    egm = EGM96('WW15MGH.DAC')
    cols, rows = dem_arr.shape
    for i in range(rows):
        for j in range(cols):
            easting, northing = retrieve_geo_coords((j, i), src_dem)
            lon, lat = utm.to_latlon(easting, northing, 43, 'U')
            geoid_height = egm.height(lon, lat)
            if dem_arr[j, i] > 0.:
                dem_arr[j, i] -= geoid_height
    return dem_arr


csv_merge(['../Field/DHUNDI_STEADY.txt', '../Field/ROHTANG.txt'])
rdem = gdal.Open('../DEM_Avg/Avg_DEM_20160119.tif')
abs_dem_arr = rel2absdem(rdem, 'Merged.csv')
write_dem_tif(abs_dem_arr, rdem, '../Abs_Dem_Avg_DEM_20160119')
print('Ellipsoidal Height DEM written...')
'''
ortho_dem_arr = ellipse2ortho(abs_dem_arr, rdem)
write_dem_tif(ortho_dem_arr, rdem, 'Ortho_Dem')
print('Orthometric Height DEM written...')
test = gdal.Open('Ortho_Dem.tif')
test_arr = test.GetRasterBand(1).ReadAsArray()
test_arr = test_arr[test_arr > 0]
print('Min=', np.min(test_arr), 'Max=', np.max(test_arr))
'''

