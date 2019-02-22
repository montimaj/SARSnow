import numpy as np
import gdal

MEAN_INC_ANGLE = (38.072940826416016 + 39.38078689575195 + 38.10858917236328 + 39.38400650024414) / 4.
NO_DATA_VALUE = -32768


def padwithzero(vector, pad_width, *args, **kwargs):
    """
    Apply zero padding
    Main author: Abhisek Maiti
    Modified by: Sayantan Majumdar
    :param vector: Input vector
    :param pad_width: Pad width to consider
    :return: Vector padded with desired number of zeros
    """

    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def derive_slope(dem_arr, incidence_angle, cell_size=12.5):
    """
    Calculate slopes in x and y directions, LIA, and Orientation Angle
    Main author: Abhisek Maiti
    Modified by: Sayantan Majumdar
    :param dem_arr: DEM Array
    :param incidence_angle: Incidence angle in radians
    :param cell_size: Pixel spacing to consider
    :return: Tuple of arrays containing slopes in x and y directions, LIA (degrees), and Orientation Angle (degrees)
    """

    dw = 1
    row, column = dem_arr.shape
    dx_arr = np.zeros(shape=(row, column))
    dy_arr = np.zeros(shape=(row, column))
    lia_arr = np.zeros(shape=(row, column))
    or_arr = np.zeros(shape=(row, column))
    dem_arr = np.lib.pad(dem_arr, dw, padwithzero).astype(float)
    if row >= ((2 * dw) + 1) and column >= ((2 * dw) + 1):
        for k in range(dw, row + 1):
            for j in range(dw, column + 1):
                block = dem_arr[k - dw:k + dw + 1, j - dw:j + dw + 1]
                if not np.isnan(block).any():
                    a = block[0][0]
                    b = block[0][1]
                    c = block[0][2]
                    d = block[1][0]
                    f = block[1][2]
                    g = block[2][0]
                    h = block[2][1]
                    i = block[2][2]
                    slope_x = ((c + (2 * f) + i) - (a + (2 * d) + g)) / (8 * cell_size)
                    slope_y = ((g + (2 * h) + i) - (a + (2 * b) + c)) / (8 * cell_size)
                    orientation = slope_x / ((-1 * slope_y * np.cos(incidence_angle)) + np.sin(incidence_angle))
                    li_num = (np.cos(np.arctan(slope_x))) * np.cos(np.arctan(slope_y) - incidence_angle)
                    li_denom = np.sqrt((((np.cos(np.arctan(slope_y))) ** 2) * ((np.sin(np.arctan(slope_x))) ** 2)) +
                                       ((np.cos(np.arctan(slope_x))) ** 2))
                    li = li_num / li_denom
                    lia_arr[k - dw][j - dw] = li
                    or_arr[k - dw][j - dw] = orientation
                    dx_arr[k - dw][j - dw] = slope_x
                    dy_arr[k - dw][j - dw] = slope_y
                else:
                    lia_arr[k - dw][j - dw] = np.nan
                    or_arr[k - dw][j - dw] = np.nan
                    dx_arr[k - dw][j - dw] = np.nan
                    dy_arr[k - dw][j - dw] = np.nan
    return dx_arr[:-1, :-1], dy_arr[:-1, :-1], np.rad2deg(lia_arr[:-1, :-1]), np.rad2deg(or_arr[:-1, :-1])


def get_image_array(img_file, set_no_data=True):
    """
    Read real numpy arrays from file
    :param set_no_data: Set False to not set nan values
    :param img_file: GDAL reference file
    :return: Numpy array with nan set accordingly
    """

    band = img_file.GetRasterBand(1)
    no_data_value = band.GetNoDataValue()
    arr = band.ReadAsArray()
    if set_no_data:
        arr[arr == no_data_value] = np.nan
    return arr


def write_file(arr, src_file, outfile='test', no_data_value=NO_DATA_VALUE):
    """
    Write image files in TIF format
    :param arr: Image array to write
    :param src_file: Original image file for retrieving affine transformation parameters
    :param outfile: Output file path
    :param no_data_value: No data value to be set
    :return: None
    """

    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(outfile + ".tif", arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    out.SetProjection(src_file.GetProjection())
    out.SetGeoTransform(src_file.GetGeoTransform())
    out.GetRasterBand(1).SetNoDataValue(no_data_value)
    arr[np.isnan(arr)] = no_data_value
    out.GetRasterBand(1).WriteArray(arr)
    out.FlushCache()


def calc_lia(dem_file, out_file):
    """
    Calculate local incidence angle (LIA)
    :param dem_file: Input DEM file
    :param out_file: Output LIA file
    :return: None
    """

    dem_file = gdal.Open(dem_file)
    dem_arr = get_image_array(dem_file)
    lia_arr = derive_slope(dem_arr, incidence_angle=np.deg2rad(MEAN_INC_ANGLE))[2]
    write_file(lia_arr, dem_file, outfile=out_file)


print('Original ALOS DEM...')
calc_lia('Inputs/Alos_Clip.tif', out_file='Outputs/Alos_LIA')
print('\nCorrected ALOS DEM...')
calc_lia('Outputs/Alos_DEM_Corr.tif', out_file='Outputs/Alos_LIA_Corr')
