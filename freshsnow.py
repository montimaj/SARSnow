from osgeo import gdal
import numpy as np
import affine
import cv2

EFF_AIR = 1.0005
EFF_ICE = 3.18
ICE_DENSITY = 0.917 # gm/cc
SNOW_DENSITY = 0.13 # gm/cc FN = 0.12, AN = 0.14
WAVELENGTH = 3.10880853
NO_DATA_VALUE = 0.


def get_depolarisation_factor(axial_ratio, shape):
    depolarisation_factorx = depolarisation_factory = depolarisation_factorz = 1/3.
    if shape == 'o':
        eccentricity = np.sqrt(axial_ratio ** 2 - 1)
        depolarisation_factorz = (1 + eccentricity**2) * (eccentricity- np.arctan(eccentricity)) / eccentricity ** 3
        depolarisation_factorx = depolarisation_factory = 0.5 * (1 - depolarisation_factorz)
    elif shape == 'p':
        eccentricity = np.sqrt(1 - axial_ratio ** 2)
        depolarisation_factorx = ((1 - eccentricity ** 2) *
                                  (np.log((1 + eccentricity) / (1 - eccentricity))
                                   - 2 * eccentricity)) / (2 * eccentricity ** 3)
        depolarisation_factory = depolarisation_factorz = 0.5 * (1 - depolarisation_factorx)
    return depolarisation_factorx, depolarisation_factory, depolarisation_factorz


def get_effective_permittivity(fvol, depolarisation_factor):
    eff_diff = EFF_ICE - EFF_AIR
    eff = EFF_AIR * (1 + fvol * eff_diff/(EFF_AIR + (1 - fvol) * depolarisation_factor * eff_diff))
    return eff


def retrieve_pixel_coords(geo_coord, data_source):
    x, y = geo_coord[0], geo_coord[1]
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    return px, py


def write_tif(arr, src_file, outfile='test'):
    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(outfile + '.tif', arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    out.SetProjection(src_file.GetProjection())
    out.SetGeoTransform(src_file.GetGeoTransform())
    out.GetRasterBand(1).SetNoDataValue(NO_DATA_VALUE)
    out.GetRasterBand(1).WriteArray(arr)
    out.FlushCache()


def get_subset_image(image_arr, pos, size):
    py, px = pos
    ny1, ny2, nx1, nx2 = py, py + size, px, px + size
    if ny1 - size >= 0:
        ny1 -= size
    if nx1 - size >= 0:
        nx1 -= size
    if ny2 > image_arr.shape[0]:
        ny2 = py
    if nx2 > image_arr.shape[1]:
        nx2 = px
    return image_arr[ny1: ny2, nx1: nx2]


def is_fresh_snow_neighborhood(cpd_data, pos, size):
    fs_neighbor = get_subset_image(cpd_data, pos, size)
    return len(fs_neighbor[fs_neighbor > 0]) > 0


def cpd2freshsnow(cpd_file_tdx, cpd_file_tsx, layover_file, outfile, neighborhood_size=11, axial_ratio=2, shape='o'):
    print('LOADING FILES TO MEMORY...')

    cpd_file_tdx = gdal.Open(cpd_file_tdx)
    cpd_file_tsx = gdal.Open(cpd_file_tsx)
    layover_file = gdal.Open(layover_file)
    cpd_tdx = cpd_file_tdx.GetRasterBand(1).ReadAsArray()
    cpd_tsx = cpd_file_tsx.GetRasterBand(1).ReadAsArray()
    lia_tdx = cpd_file_tdx.GetRasterBand(3).ReadAsArray()
    lia_tsx = cpd_file_tsx.GetRasterBand(3).ReadAsArray()
    layover_arr = layover_file.GetRasterBand(1).ReadAsArray()

    print('ALL FILES LOADED... CALCULATING PARAMETERS...')

    cpd_data = (cpd_tdx + cpd_tsx) / 2.
    #cpd_data = cv2.bilateralFilter(cpd_data, 9, 100, 100)
    lia_data = (lia_tdx + lia_tsx) / 2.
    #lia_data = cv2.bilateralFilter(lia_data, 9, 100, 100)
    fvol = SNOW_DENSITY/ICE_DENSITY
    depolarisation_factors = get_depolarisation_factor(axial_ratio, shape)
    print('DEPOLARISATION FACTORS = ', depolarisation_factors)
    effx = get_effective_permittivity(fvol, depolarisation_factors[0])
    effy = get_effective_permittivity(fvol, depolarisation_factors[1])
    effz = get_effective_permittivity(fvol, depolarisation_factors[2])

    print('PIXELWISE COMPUTATION STARTED...')

    cols, rows = cpd_data.shape
    fresh_sd = np.zeros(cpd_data.shape).astype(np.float32)
    for i in range(rows):
        for j in range(cols):
            cpd = cpd_data[j, i]
            if layover_arr[j, i] == 0. and (cpd > 0 or (not np.isnan(cpd) and
                                                        is_fresh_snow_neighborhood(cpd_data, (j, i), neighborhood_size))):
                sin_lia_sq = np.sin(lia_data[j, i]) ** 2
                effH = effx ** 2
                effV = effy * np.cos(lia_data[j, i]) ** 2 + effz * sin_lia_sq
                xeta_diff = np.sqrt(effV - sin_lia_sq) - np.sqrt(effH - sin_lia_sq)
                fresh_sd[j, i] = np.abs(np.float32(cpd * WAVELENGTH / (4 * np.pi * xeta_diff)))
            else:
                fresh_sd[j, i] = np.float32(NO_DATA_VALUE)

    print('WRITING UNFILTERED IMAGE...')

    write_tif(fresh_sd, cpd_file_tdx, outfile)

def get_image_stats(image_arr):
    return np.min(image_arr), np.max(image_arr), np.mean(image_arr), np.var(image_arr)


def validate_fresh_snow(fsd_file, geocoords, validation_file, nsize=11):
    fsd_file = gdal.Open(fsd_file)
    #geocoords = utm.from_latlon(geocoords[0], geocoords[1])[:2]
    px, py = retrieve_pixel_coords(geocoords, fsd_file)
    fsd_arr = fsd_file.GetRasterBand(1).ReadAsArray()
    print('IMAGE STATS...')
    print('STUDY AREA FRESH SNOW DEPTH (min, max, mean, var) = ', get_image_stats(fsd_arr[fsd_arr > 0]))
    fsd_dhundi = get_subset_image(fsd_arr, (py, px), nsize)
    np.savetxt(validation_file, fsd_dhundi)
    min_fsd, max_fsd, mean_fsd, var_fsd = get_image_stats(fsd_dhundi[fsd_dhundi > 0])
    max_pos = np.where(fsd_arr == max_fsd) # ground value = 3.4
    print('Pixels = ', (py, px), 'Max_pos = ', max_pos)
    print('DHUNDI FRESH SNOW DEPTH (min, max, mean, var) = ', min_fsd, max_fsd, mean_fsd, var_fsd)


def filter_image(image_file, outfile):
    image_file = gdal.Open(image_file)
    img_arr = image_file.GetRasterBand(1).ReadAsArray()
    print('\nWRITING BILATERAL FILTERED IMAGE...')
    img_arr[img_arr != NO_DATA_VALUE] = np.array(cv2.bilateralFilter(img_arr[img_arr != NO_DATA_VALUE],
                                                                     d=3, sigmaColor=2, sigmaSpace=15)).flat
    write_tif(img_arr, image_file, outfile)


cpd2freshsnow('CoSSC_TDX.tif', 'CoSSC_TSX.tif', 'avg_layover.tif', 'fsd')
print('UNFILTERED IMAGE VALIDATION...')
validate_fresh_snow('Fresh_Snow/Out/fsd.tif', (700097.9845, 3581763.7627), 'val.csv')
filter_image('Fresh_Snow/Out/fsd.tif', 'Fresh_Snow/Out/fsd_flt')
print('FILTERED IMAGE VALIDATION...')
validate_fresh_snow('Fresh_Snow/Out/fsd_flt.tif', (700097.9845, 3581763.7627), 'val_flt.csv', 11)
