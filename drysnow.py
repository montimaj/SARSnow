import numpy as np
import affine
from osgeo import gdal

C_BAND_WAVELENGTH = 5.550415814663357
MEAN_DENSITY = 0.38285714285714284
NO_DATA_VALUE = 0


def write_tif(arr, src_file, outfile='test', no_data_value=NO_DATA_VALUE):
    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(outfile + '.tif', arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    out.SetProjection(src_file.GetProjection())
    out.SetGeoTransform(src_file.GetGeoTransform())
    out.GetRasterBand(1).SetNoDataValue(no_data_value)
    out.GetRasterBand(1).WriteArray(arr)
    out.FlushCache()


def get_ensemble_window(image_arr, index, wsize):
    startx = index[0] - wsize[0]
    starty = index[1] - wsize[1]
    if startx < 0:
        startx = 0
    if starty < 0:
        starty = 0
    endx = index[0] + wsize[0] + 1
    endy = index[1] + wsize[1] + 1
    limits = image_arr.shape[0] + 1, image_arr.shape[1] + 1
    if endx > limits[0] + 1:
        endx = limits[0] + 1
    if endy > limits[1] + 1:
        endy = limits[1] + 1
    return image_arr[startx: endx, starty: endy]


def get_ensemble_avg(image_arr, wsize):
    print('PERFORMING ENSEMBLE AVERAGING...')
    emat = np.full_like(image_arr, NO_DATA_VALUE, dtype=np.float32)
    for index, value in np.ndenumerate(image_arr):
        if not np.isnan(value):
            ensemble_window = get_ensemble_window(image_arr, index, wsize)
            emat[index] = np.mean(ensemble_window[~np.isnan(ensemble_window)])
            print(index, emat[index])
    return emat


def filter_image(image_arr, wsize):
    image_arr[image_arr == NO_DATA_VALUE] = np.nan
    print('\nWRITING FILTERED IMAGE...')
    flt_arr = get_ensemble_avg(image_arr, wsize)
    flt_arr[np.isnan(flt_arr)] = NO_DATA_VALUE
    return flt_arr


def retrieve_pixel_coords(geo_coord, data_source):
    x, y = geo_coord[0], geo_coord[1]
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    return px, py


def get_image_stats(image_arr):
    return np.min(image_arr), np.max(image_arr), np.mean(image_arr), np.var(image_arr)


def validate_dry_snow(dsd_file, geocoords, nsize=(11, 11)):
    dsd_file = gdal.Open(dsd_file)
    #geocoords = utm.from_latlon(geocoords[0], geocoords[1])[:2]
    px, py = retrieve_pixel_coords(geocoords, dsd_file)
    dsd_arr = dsd_file.GetRasterBand(1).ReadAsArray()
    print('IMAGE STATS...')
    print('STUDY AREA FRESH SNOW DEPTH (min, max, mean, var) = ', get_image_stats(dsd_arr[dsd_arr != NO_DATA_VALUE]))
    dsd_dhundi = get_ensemble_window(dsd_arr, (py, px), nsize)
    min_fsd, max_fsd, mean_dsd, var_fsd = get_image_stats(dsd_dhundi[dsd_dhundi != NO_DATA_VALUE])
    max_pos = np.where(dsd_arr == max_fsd) # ground value = 52 cm
    print('Pixels = ', (py, px), 'Max_pos = ', max_pos)
    print('DHUNDI FRESH SNOW DEPTH (min, max, mean, var) = ', min_fsd, max_fsd, mean_dsd, var_fsd)


# input_image = gdal.Open('dsd_data.tif')
# unw_phase = input_image.GetRasterBand(1).ReadAsArray()
# lia_data = input_image.GetRasterBand(2).ReadAsArray()
# #print('Filtering unwrapped phase...')
# #write_tif(unw_phase, input_image, 'unw_flt')
#
# #unw_image = gdal.Open('unw_flt.tif')
# #unw_phase = unw_image.GetRasterBand(1).ReadAsArray()
# epsilon_snow = 1 + 1.5995 * MEAN_DENSITY + 1.861 * MEAN_DENSITY ** 3
# del_eta = np.cos(lia_data) - np.sqrt(epsilon_snow - np.sin(lia_data) ** 2)
# dry_snow_arr = np.abs(C_BAND_WAVELENGTH * unw_phase / (4 * np.pi * del_eta))
# print('Filtering dry snow depth...')
# dry_snow_flt = filter_image(dry_snow_arr, (51, 51))
# write_tif(dry_snow_flt, input_image, 'dry_snow_depth')
# #dry_snow_flt = filter_image(dry_snow_arr, (1, 1))
# #write_tif(dry_snow_flt, input_image, 'dry_snow_depth_sw_1')

validate_dry_snow('dry_snow_depth.tif', (700089.771, 3581794.5556), (11, 11))