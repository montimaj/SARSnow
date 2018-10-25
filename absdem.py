from osgeo import gdal
import numpy as np
import affine


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


def write_dem(dem_arr, projection, geotransform):
    driver = gdal.GetDriverByName("GTiff")
    absdem = driver.Create('Abs_Dem.tif', dem_arr.shape[1], dem_arr.shape[0], 1, gdal.GDT_Float32)
    absdem.SetProjection(projection)
    absdem.SetGeoTransform(geotransform)
    absdem.GetRasterBand(1).SetNoDataValue(0.)
    absdem.GetRasterBand(1).WriteArray(dem_arr)
    absdem.FlushCache()


def rel2absdem(rdem, gcps):
    rdem = gdal.Open(rdem)
    gcps = open(gcps, 'r')
    ground_points = gcps.readlines()
    rdem_arr = rdem.GetRasterBand(1).ReadAsArray()
    rdem_arr[rdem_arr == 0.] = np.nan
    zero_abs_height = []
    for gcp in ground_points:
        val = gcp.split(',')
        x = float(val[1])
        y = float(val[2])
        z = float(val[3])
        px, py = retrieve_pixel_coords((x, y), rdem)
        rel_height = rdem_arr[py, px]
        if rel_height <= 0:
            zero_abs_height.append(z - rel_height)
        else:
            zero_abs_height.append(rel_height - z)
    zero_abs_height = np.mean(zero_abs_height)
    rdem_arr += zero_abs_height
    rdem_arr[rdem_arr == np.nan] = 0.
    write_dem(rdem_arr, rdem.GetProjection(), rdem.GetGeoTransform())


csv_merge(['../Field/DHUNDI_STEADY.txt', '../Field/ROHTANG.txt'])
rel2absdem('rel_dem.tif', 'Merged.csv')
