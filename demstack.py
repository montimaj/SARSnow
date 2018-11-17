from osgeo import gdal
import numpy as np
import os
import glob
import cv2
from collections import defaultdict

NO_DATA_VALUE = -32768.0


def read_images(path, imgformat='*.tif'):

    """
    Function to read images having provided extension.
    :param path: Directory to read.
    :param imgformat: Image format.
    :return: Dictionary of images where the key is an image year and the value is a GDAL Image Object.
    """

    print("Reading images...")
    images = defaultdict(lambda: [])
    files = os.path.join(path, imgformat)  # create file pattern
    for file in glob.glob(files):  # for each file matching the above pattern
        pos = file.rfind('_') + 1  # get the last file separator position
        date = file[pos: pos + 8]  # extract date from file name
        images[date].append(gdal.Open(file))  # add GDAL object for the specific year
    print("Finished reading")
    return images


def get_averaged_dem(dem_files, imgformat='*.tif'):
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


def write_dem_tif(dem_arr_list, src_file, outfile='test', no_data_value=NO_DATA_VALUE):
    driver = gdal.GetDriverByName("GTiff")
    num_bands = len(dem_arr_list)
    absdem = driver.Create(outfile + '.tif', dem_arr_list[0].shape[1], dem_arr_list[1].shape[0], num_bands, gdal.GDT_Float32)
    absdem.SetProjection(src_file.GetProjection())
    absdem.SetGeoTransform(src_file.GetGeoTransform())
    for i in range(num_bands):
        absdem.GetRasterBand(i+1).SetNoDataValue(no_data_value)
        absdem.GetRasterBand(i+1).WriteArray(dem_arr_list[i])
    absdem.FlushCache()


def get_min_error_key(err_file_dict, pos):
    error_dict = {}
    for date, err_arr in err_file_dict.items():
        error = err_arr[pos[0], pos[1]]
        if error != NO_DATA_VALUE:
            error_dict[date] = error
    if error_dict:
        return min(error_dict, key=error_dict.get)
    return NO_DATA_VALUE


def generate_optimized_dem(dem_files_dict, dem_band=1, err_band=4):
    dem_val_dict = {}
    dem_err_dict = {}
    ncols, nrows = 0, 0
    for date, files in dem_files_dict.items():
        val_arr = files[0].GetRasterBand(dem_band).ReadAsArray()
        err_arr = files[0].GetRasterBand(err_band).ReadAsArray()
        val_arr[val_arr == 0] = NO_DATA_VALUE
        dem_val_dict[date] = val_arr
        dem_err_dict[date] = err_arr
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
    write_dem_tif(opt_dem, dem_files_dict[src_file_key], 'dem')
    write_dem_tif(minimized_error, dem_files_dict[src_file_key], 'error')


def filter_dem(dem_file, outfile):
    dem_file = gdal.Open(dem_file)
    dem_arr = dem_file.GetRasterBand(1).ReadAsArray()
    print('Using Bilateral Filter....')
    dem_arr[dem_arr != NO_DATA_VALUE] = np.array(cv2.blur(dem_arr[dem_arr != NO_DATA_VALUE], (99, 99))).flat
    write_dem_tif(dem_arr, dem_file, outfile)


get_averaged_dem('Rel_DEM_Tifs', '*.tif')
#dem_files_dict = read_images('Clipped_DEM')
#generate_optimized_dem(dem_files_dict)
#filter_dem('/home/iirs/THESIS/SnowSAR/Wet_Snow_Stack/dem.tif', '/home/iirs/THESIS/SnowSAR/Wet_Snow_Stack/dem_flt')