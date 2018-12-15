from osgeo import gdal
import numpy as np
import os
import glob
from collections import defaultdict

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


def read_images(path, imgformat='*.tif', makelist=False):
    print("Reading images...")
    if makelist:
        images = defaultdict(lambda: [])
    else:
        images = {}
    files = os.path.join(path, imgformat)
    for file in glob.glob(files):
        pos = file.rfind('_')
        if makelist:
            key = file[pos + 1: pos + 8]
            images[key].append(gdal.Open(file))
        else:
            key = file[pos - 2: file.find('.')]
            images[key] = gdal.Open(file)
    print("Finished reading")
    return images


def create_error_maps(dem_files_dict, dem_band=1, ref_dem_band=2, outdir='DEM_Errors'):
    for key, dem_file in dem_files_dict.items():
        tdm_arr = dem_file.GetRasterBand(dem_band).ReadAsArray()
        ref_arr = dem_file.GetRasterBand(ref_dem_band).ReadAsArray()
        tdm_arr[tdm_arr == NO_DATA_VALUE] = np.nan
        ref_arr[ref_arr == NO_DATA_VALUE] = np.nan
        print('Calculating error map for:', key)
        err_arr = np.abs(tdm_arr - ref_arr)
        err_arr[np.isnan(err_arr)] = NO_DATA_VALUE
        write_dem_tif([tdm_arr, err_arr], dem_file, outdir + '/DEM_Error_' + key)


def write_dem_tif(dem_arr_list, src_file, outfile='test', no_data_value=NO_DATA_VALUE):
    driver = gdal.GetDriverByName("GTiff")
    num_bands = len(dem_arr_list)
    absdem = driver.Create(outfile + '.tif', dem_arr_list[0].shape[1], dem_arr_list[0].shape[0], num_bands, gdal.GDT_Float32)
    absdem.SetProjection(src_file.GetProjection())
    absdem.SetGeoTransform(src_file.GetGeoTransform())
    for i in range(num_bands):
        absdem.GetRasterBand(i + 1).SetNoDataValue(no_data_value)
        absdem.GetRasterBand(i + 1).WriteArray(dem_arr_list[i])
    absdem.FlushCache()


def get_min_error_key(err_file_dict, pos):
    error_dict = {}
    for key, err_arr in err_file_dict.items():
        error = err_arr[pos[0], pos[1]]
        if error != NO_DATA_VALUE:
            error_dict[key] = error
    if error_dict:
        return min(error_dict, key=error_dict.get)
    return NO_DATA_VALUE


def generate_optimized_dem(dem_files_dict, dem_band=1, err_band=2):
    dem_val_dict = {}
    dem_err_dict = {}
    nrows, ncols = 0, 0
    for key, file in dem_files_dict.items():
        val_arr = file.GetRasterBand(dem_band).ReadAsArray()
        err_arr = file.GetRasterBand(err_band).ReadAsArray()
        val_arr[val_arr == 0] = NO_DATA_VALUE
        dem_val_dict[key] = val_arr
        dem_err_dict[key] = err_arr
        nrows, ncols = val_arr.shape
    opt_dem = np.zeros((nrows, ncols))
    minimized_error = np.zeros((nrows, ncols))
    opt_dem.fill(NO_DATA_VALUE)
    minimized_error.fill(NO_DATA_VALUE)
    for i in range(nrows):
        for j in range(ncols):
            min_err_key = get_min_error_key(dem_err_dict, (i, j))
            if min_err_key != NO_DATA_VALUE:
                opt_dem[i, j] = dem_val_dict[min_err_key][i, j]
                minimized_error[i, j] = dem_err_dict[min_err_key][i, j]
        print('At row:', i)
    src_file_key = list(dem_files_dict.keys())[0]
    write_dem_tif([opt_dem], dem_files_dict[src_file_key], 'dem')
    write_dem_tif([minimized_error], dem_files_dict[src_file_key], 'error')


dem_files_dict = read_images('Clipped_DEMs')
create_error_maps(dem_files_dict)
dem_files_dict = read_images('DEM_Errors')
generate_optimized_dem(dem_files_dict)