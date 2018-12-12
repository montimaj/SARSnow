from osgeo import gdal
import numpy as np
import affine
#import cv2
import scipy.stats as st
#import matplotlib.pyplot as plt

EFF_AIR = 1.0005
EFF_ICE = 3.179
ICE_DENSITY = 0.917 # gm/cc
SNOW_DENSITY = 0.13 # gm/cc FN = 0.12, AN = 0.14
WAVELENGTH = 3.10880853
NO_DATA_VALUE = -32768
MEAN_INC_ANGLE = (38.072940826416016 + 39.38078689575195 + 38.10858917236328 + 39.38400650024414)/4.


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


# def retrieve_pixel_coords(geo_coord, data_source):
#     x, y = geo_coord[0], geo_coord[1]
#     forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
#     reverse_transform = ~forward_transform
#     px, py = reverse_transform * (x, y)
#     px, py = int(px + 0.5), int(py + 0.5)
#     return px, py


def write_tif(arr, src_file, outfile='test', no_data_value=NO_DATA_VALUE):
    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(outfile + '.tif', arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    out.SetProjection(src_file.GetProjection())
    out.SetGeoTransform(src_file.GetGeoTransform())
    out.GetRasterBand(1).SetNoDataValue(no_data_value)
    out.GetRasterBand(1).WriteArray(arr)
    out.FlushCache()


def get_gaussian_kernel(ksize, nsig=3):
    interval = (2 * nsig + 1.) / ksize[0]
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., ksize[0] + 1)
    interval = (2 * nsig + 1.) / ksize[1]
    y = np.linspace(-nsig - interval / 2., nsig + interval / 2., ksize[1] + 1)
    k1 = np.diff(st.norm.cdf(x))
    k2 = np.diff(st.norm.cdf(y))
    kernel_raw = np.sqrt(np.outer(k1, k2))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


# def is_fresh_snow_neighborhood(cpd_data, pos, size):
#     fs_neighbor = get_ensemble_window(cpd_data, pos, size)
#     return len(fs_neighbor[fs_neighbor > 0]) > 0

def get_ensemble_avg(image_arr, wsize=(10, 10), nsig=3):
    print('PERFORMING ENSEMBLE AVERAGING...')
    image_arr = np.transpose(image_arr)
    max_x, max_y = image_arr.shape
    max_x += 1
    max_y += 1
    emat = np.full_like(image_arr, NO_DATA_VALUE, dtype=np.float32)
    for index, value in np.ndenumerate(image_arr):
        if not np.isnan(value):
            startx = index[0] - wsize[0]
            starty = index[1] - wsize[1]
            if startx < 0:
                startx = 0
            if starty < 0:
                starty = 0
            endx = index[0] + wsize[0] + 1
            endy = index[1] + wsize[1] + 1
            if endx > max_x + 1:
                endx = max_x + 1
            if endy > max_y + 1:
                endy = max_y + 1
            ensemble_window = image_arr[startx: endx, starty: endy]
            gkernel = get_gaussian_kernel(ensemble_window.shape, nsig)
            wt_values = gkernel * ensemble_window
            emat[index] = np.mean(wt_values[~np.isnan(wt_values)])
            print(index, emat[index])
    return np.transpose(emat)


def do_averaging(cpd_file_tdx, cpd_file_tsx, outfile_cpd, outfile_lia):
    print('LOADING FILES TO MEMORY...')

    cpd_file_tdx = gdal.Open(cpd_file_tdx)
    cpd_file_tsx = gdal.Open(cpd_file_tsx)
    cpd_tdx = cpd_file_tdx.GetRasterBand(1).ReadAsArray()
    cpd_tsx = cpd_file_tsx.GetRasterBand(1).ReadAsArray()
    lia_tdx = cpd_file_tdx.GetRasterBand(3).ReadAsArray()
    lia_tsx = cpd_file_tsx.GetRasterBand(3).ReadAsArray()

    print('ALL FILES LOADED... AVERAGING...')
    cpd_data = get_ensemble_avg((cpd_tdx + cpd_tsx) / 2.)
    lia_data = (lia_tdx + lia_tsx) / 2.
    print('Writing avg data...')
    write_tif(cpd_data, cpd_file_tdx, outfile_cpd)
    write_tif(lia_data, cpd_file_tdx, outfile_lia)


def cpd2freshsnow(avg_cpd_file, avg_lia_file, layover_file, outfile, axial_ratio=2, shape='o'):
    print('LOADING FILES ...')
    avg_cpd_file = gdal.Open(avg_cpd_file)
    avg_lia_file = gdal.Open(avg_lia_file)
    layover_file = gdal.Open(layover_file)
    cpd_data = avg_cpd_file.GetRasterBand(1).ReadAsArray()
    lia_data = avg_lia_file.GetRasterBand(1).ReadAsArray()
    layover_arr = layover_file.GetRasterBand(1).ReadAsArray()

    print('ALL FILES LOADED... CALCULATING PARAMETERS...')
    fvol = SNOW_DENSITY/ICE_DENSITY
    depolarisation_factors = get_depolarisation_factor(axial_ratio, shape)
    print('DEPOLARISATION FACTORS = ', depolarisation_factors)
    effx = get_effective_permittivity(fvol, depolarisation_factors[0])
    effy = get_effective_permittivity(fvol, depolarisation_factors[1])
    effz = get_effective_permittivity(fvol, depolarisation_factors[2])

    print('PIXELWISE COMPUTATION STARTED...')
    print('Mean incidence angle=', MEAN_INC_ANGLE)
    cols, rows = cpd_data.shape
    fresh_sd = np.zeros(cpd_data.shape).astype(np.float32)
    for i in range(rows):
        for j in range(cols):
            cpd = cpd_data[j, i]
            if layover_arr[j, i] == 0. and cpd > 0:
                sin_inc_sq = np.sin(lia_data[j, i]) ** 2
                effH = effx
                effV = effy * np.cos(lia_data[j, i]) ** 2 + effz * sin_inc_sq
                xeta_diff = np.sqrt(effV - sin_inc_sq) - np.sqrt(effH - sin_inc_sq)
                print('Effective permittivities=', str(effH), str(effV))
                print('Del Xeta=', str(xeta_diff))
                fresh_sd[j, i] = np.abs(np.float32(cpd * WAVELENGTH / (4 * np.pi * xeta_diff)))
            else:
                fresh_sd[j, i] = np.float32(NO_DATA_VALUE)

    print('WRITING UNFILTERED IMAGE...')
    write_tif(fresh_sd, avg_cpd_file, outfile)


# def get_image_stats(image_arr):
#     return np.min(image_arr), np.max(image_arr), np.mean(image_arr), np.var(image_arr)
#
#
# def validate_fresh_snow(fsd_file, geocoords, validation_file, nsize=11):
#     fsd_file = gdal.Open(fsd_file)
#     #geocoords = utm.from_latlon(geocoords[0], geocoords[1])[:2]
#     px, py = retrieve_pixel_coords(geocoords, fsd_file)
#     fsd_arr = fsd_file.GetRasterBand(1).ReadAsArray()
#     print('IMAGE STATS...')
#     print('STUDY AREA FRESH SNOW DEPTH (min, max, mean, var) = ', get_image_stats(fsd_arr[fsd_arr > 0]))
#     #fsd_dhundi = get_ensemble_window(fsd_arr, (py, px), nsize)
#     np.savetxt(validation_file, fsd_dhundi)
#     min_fsd, max_fsd, mean_fsd, var_fsd = get_image_stats(fsd_dhundi[fsd_dhundi > 0])
#     max_pos = np.where(fsd_arr == max_fsd) # ground value = 3.4
#     print('Pixels = ', (py, px), 'Max_pos = ', max_pos)
#     print('DHUNDI FRESH SNOW DEPTH (min, max, mean, var) = ', min_fsd, max_fsd, mean_fsd, var_fsd)
#
#
# def filter_image(image_file, outfile):
#     image_file = gdal.Open(image_file)
#     img_arr = image_file.GetRasterBand(1).ReadAsArray()
#     print('\nWRITING BILATERAL FILTERED IMAGE...')
#     img_arr[img_arr != NO_DATA_VALUE] = np.array(cv2.bilateralFilter(img_arr[img_arr != NO_DATA_VALUE],
#                                                                      d=-1, sigmaColor=2, sigmaSpace=7)).flat
#     write_tif(img_arr, image_file, outfile)


do_averaging('Out/cpd_tdx_TC.tif', 'Out/cpd_tsx_TC.tif', 'Fresh_Snow/cpd_avg', 'Fresh_Snow/lia_avg')
cpd2freshsnow('Fresh_Snow/cpd_avg.tif', 'Fresh_Snow/lia_avg.tif', 'Out/layover.tif', 'Fresh_Snow/fsd')
#print('UNFILTERED IMAGE VALIDATION...')
#validate_fresh_snow('Fresh_Snow/Out/fsd.tif', (700097.9845, 3581763.7627), 'val.csv')
#filter_image('Fresh_Snow/Out/fsd.tif', 'Fresh_Snow/Out/fsd_flt')
#print('FILTERED IMAGE VALIDATION...')
#validate_fresh_snow('Fresh_Snow/Out/fsd_flt.tif', (700097.9845, 3581763.7627), 'val_flt.csv', 11)
#gk = get_gaussian_kernel((21,11))
#plt.imshow(gk, interpolation=None)
#plt.show()