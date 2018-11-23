from osgeo import gdal
import numpy as np
import os
import glob
import cv2
from collections import defaultdict

NO_DATA_VALUE = -32768.0


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


def create_averaged_dem(dem_files, imgformat='*.tif'):
    dem_images_dict = read_images(dem_files, imgformat)
    for date, image_list in dem_images_dict.items():
        print('Averaging date: ', date)
        numbands = image_list[0].RasterCount
        srcfile = image_list[0]
        arr = defaultdict(lambda: 0)
        for band in range(numbands):
            for image in image_list:
                arr[band] += image.GetRasterBand(band + 1).ReadAsArray()
        avg_arr_list = []
        for band in arr.keys():
            avg_arr = arr[band]/len(image_list)
            avg_arr[np.isnan(avg_arr)] = NO_DATA_VALUE
            avg_arr_list.append(avg_arr)
        print('Writing file...')
        write_dem_tif(avg_arr_list, srcfile, 'DEM_Avg/Avg_DEM_' + date)


def create_error_maps(dem_files_dict, dem_band=1, ref_dem_band=2, outdir='DEM_Errors'):
    for key, dem_file in dem_files_dict.items():
        tdm_arr = dem_file.GetRasterBand(dem_band).ReadAsArray()
        ref_arr = dem_file.GetRasterBand(ref_dem_band).ReadAsArray()
        err_arr = np.zeros(tdm_arr.shape)
        err_arr.fill(NO_DATA_VALUE)
        print('Calculating error map for:', key)
        for i in range(err_arr.shape[1]):
            for j in range(err_arr.shape[0]):
                if tdm_arr[j, i] != NO_DATA_VALUE and ref_arr[j, i] != NO_DATA_VALUE:
                    err_arr[j, i] = np.abs(tdm_arr[j, i] - ref_arr[j, i])
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
    ncols, nrows = 0, 0
    for key, file in dem_files_dict.items():
        val_arr = file.GetRasterBand(dem_band).ReadAsArray()
        err_arr = file.GetRasterBand(err_band).ReadAsArray()
        val_arr[val_arr == 0] = NO_DATA_VALUE
        dem_val_dict[key] = val_arr
        dem_err_dict[key] = err_arr
        ncols, nrows = val_arr.shape
    opt_dem = np.zeros((ncols, nrows))
    minimized_error = np.zeros((ncols, nrows))
    opt_dem.fill(NO_DATA_VALUE)
    minimized_error.fill(NO_DATA_VALUE)
    for i in range(nrows):
        for j in range(ncols):
            min_err_key = get_min_error_key(dem_err_dict, (j, i))
            if min_err_key != NO_DATA_VALUE:
                opt_dem[j, i] = dem_val_dict[min_err_key][j, i]
                minimized_error[j, i] = dem_err_dict[min_err_key][j, i]
        print('At row:', i)
    src_file_key = list(dem_files_dict.keys())[0]
    write_dem_tif([opt_dem], dem_files_dict[src_file_key], 'dem')
    write_dem_tif([minimized_error], dem_files_dict[src_file_key], 'error')


def filter_dem(dem_file, outfile):
    dem_file = gdal.Open(dem_file)
    dem_arr = dem_file.GetRasterBand(1).ReadAsArray()
    print('Using Bilateral Filter....')
    dem_arr[dem_arr != NO_DATA_VALUE] = np.array(cv2.blur(dem_arr[dem_arr != NO_DATA_VALUE], (99, 99))).flat
    write_dem_tif(dem_arr, dem_file, outfile)


#create_averaged_dem('Rel_DEM_Tifs', '*.tif')
dem_files_dict = read_images('Clipped_DEM')
create_error_maps(dem_files_dict)
dem_files_dict = read_images('DEM_Errors')
generate_optimized_dem(dem_files_dict)
#filter_dem('/home/iirs/THESIS/SnowSAR/Wet_Snow_Stack/dem.tif', '/home/iirs/THESIS/SnowSAR/Wet_Snow_Stack/dem_flt')