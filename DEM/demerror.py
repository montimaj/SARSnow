import pandas as pd
import gdal
import numpy as np
import affine


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
    px, py = int(px + 0.5), int(py + 0.5)
    return px, py


def calculate_errors(dem_arr, dem_file, point_list):
    """
    Calculate DEM errors
    :param dem_arr: DEM array to validate
    :param dem_file: Original GDAL reference having affine transformation parameters
    :param point_list: List of tuples containing N, X, Y, and Z values
    :return None
    """

    N = []
    X = []
    Y = []
    E = []
    for point in point_list:
        name = point[0]
        x = point[1]
        y = point[2]
        z = point[3]
        px, py = retrieve_pixel_coords((x, y), dem_file)
        error = dem_arr[py, px] - z
        N.append(name)
        X.append(x)
        Y.append(y)
        E.append(error)
    df = pd.DataFrame({'N': N, 'X': X, 'Y': Y, 'E': E})
    df.to_csv('DEM_Error.csv', index=False)


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
dem_arr = get_image_array(alos_dem)
rp = create_point_list(rohtang_df)
dp = create_point_list(dhundi_df)
all_points = rp + dp
calculate_errors(dem_arr, alos_dem, all_points)