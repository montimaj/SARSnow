import pandas as pd
import gdal
import numpy as np
import affine


NO_DATA_VALUE = -32768


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


def retrieve_pixel_coords(geo_coord, data_source):
    """
    Get pixels coordinates from geo-coordinates
    :param geo_coord: Geo-cooridnate tuple
    :param data_source: Original GDAL reference having affine transformation parameters
    :return: Pixel coordinates in x and y direction (should be reversed in the caller function to get the actual pixel
    position)
    """

    x, y = geo_coord[0], geo_coord[1]
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    # px, py = int(px + 0.5), int(py + 0.5)
    px, py = int(np.floor(px)), int(np.floor(py))
    return px, py


def write_file(arr, src_file, outfile='test', no_data_value=NO_DATA_VALUE):
    """
    Write image files in TIF format
    :param arr: Image array to write
    :param src_file: Original image file for retrieving affine transformation parameters
    :param outfile: Output file path
    :param no_data_value: No data value to be set
    :param is_complex: If true, write complex image array in two separate bands
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


def calculate_errors(dem_arr, lia_arr, lia_arr_corr, dem_file, point_list, correct_dem=True):
    """
    Calculate DEM errors
    :param dem_arr: DEM array to validate
    :param dem_file: Original GDAL reference having affine transformation parameters
    :param lia_arr: Uncorrected Local Incidence Angle
    :param lia_arr_corr: Corrected Local Incidence Angle
    :param point_list: List of tuples containing N, X, Y, and Z values
    :param correct_dem: Set true to write correct DEM
    :return None
    """

    N = []
    X = []
    Y = []
    E1 = []
    E2 = []
    if correct_dem:
        dem_corr = dem_arr.copy()
    for point in point_list:
        name = point[0]
        x = point[1]
        y = point[2]
        z = point[3]
        px, py = retrieve_pixel_coords((x, y), dem_file)
        if correct_dem:
            dem_corr[py, px] = z
        error1 = dem_arr[py, px] - z
        if np.isnan(error1):
            error1 = NO_DATA_VALUE
        error2 = lia_arr[py, px] - lia_arr_corr[py, px]
        if np.isnan(error2):
            error2 = NO_DATA_VALUE
        N.append(name)
        X.append(x)
        Y.append(y)
        E1.append(np.round(error1, 2))
        E2.append(np.round(error2, 2))

    df = pd.DataFrame({'N': N, 'X': X, 'Y': Y, 'E1': E1, 'E2': E2})
    df.to_csv('DEM_Error.csv', index=False)
    if correct_dem:
        write_file(dem_corr.copy(), dem_file, outfile='Outputs/Alos_DEM_Corr')


def create_point_list(df):
    """
    Create list of tuples containing N, X, Y, and Z values
    :param df: Pandas dataframe
    :return: List of tuples
    """

    points = [(row['N'], row['X'], row['Y'], row['Z']) for index, row in df.iterrows()]
    return points


rohtang_df = pd.read_csv('Inputs/rohtang.csv')
dhundi_df = pd.read_csv('Inputs/dhundi.csv')
alos_dem = gdal.Open('Inputs/Alos_Clip.tif')
lia_file1 = gdal.Open('Outputs/Alos_LIA.tif')
lia_file2 = gdal.Open('Outputs/Alos_LIA_Corr.tif')
dem_arr = get_image_array(alos_dem)
lia_arr1 = get_image_array(lia_file1)
lia_arr2 = get_image_array(lia_file2)
rp = create_point_list(rohtang_df)
dp = create_point_list(dhundi_df)
all_points = rp + dp
calculate_errors(dem_arr, lia_arr1, lia_arr2, alos_dem, all_points, correct_dem=False)