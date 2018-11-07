from osgeo import gdal
import numpy as np
import affine
import utm
#import cv2

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


def cpd2freshsnow(cpd_file_tdx, cpd_file_tsx, layover_file, outfile, axial_ratio=2, shape='o'):
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
    lia_data = (lia_tdx + lia_tsx) / 2.
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
            if cpd > 0 and layover_arr[j, i] == 0.:
                sin_lia_sq = np.sin(lia_data[j, i]) ** 2
                effH = effx ** 2
                effV = effy * np.cos(lia_data[j, i]) ** 2 + effz * sin_lia_sq
                xeta_diff = np.sqrt(effV - sin_lia_sq) - np.sqrt(effH - sin_lia_sq)
                fresh_sd[j, i] = np.abs(np.float32(cpd * WAVELENGTH / (4 * np.pi * xeta_diff)))
            else:
                fresh_sd[j, i] = np.float32(NO_DATA_VALUE)

    #print('INVOKING BILATERAL FILTER...')

    #fresh_sd[fresh_sd != -32767.0] = cv2.bilateralFilter(, 9, 100, 100)

    print('WRITING UNFILTERED IMAGE...')

    write_tif(fresh_sd, cpd_file_tdx, outfile)

    #print('WRITING FILTERED IMAGE...')

    #write_tif(fresh_sd_flt, cpd_file_tdx, outfile + '_flt')


def validate_fresh_snow(fsd_file, geocoords):
    fsd_file = gdal.Open(fsd_file)
    #geocoords = utm.from_latlon(geocoords[0], geocoords[1])[:2]
    px, py = retrieve_pixel_coords(geocoords, fsd_file)
    fsd_arr = fsd_file.GetRasterBand(1).ReadAsArray()
    fsd_dhundi = fsd_arr[py - 11: py + 11, px - 11: px + 11]
    np.savetxt('validation.csv', fsd_dhundi)
    mean_fsd, max_fsd = np.mean(fsd_dhundi[fsd_dhundi > 0]), np.max(fsd_dhundi[fsd_dhundi > 0])
    max_pos = np.where(fsd_arr == max_fsd) # ground value = 3.4
    print('Pixels = ', (py, px), 'Max_pos = ', max_pos)
    print('DHUNDI FRESH SNOW DEPTH = ', mean_fsd, max_fsd)


cpd2freshsnow('CoSSC_TDX.tif', 'CoSSC_TSX.tif', 'avg_layover.tif', 'fsd')
validate_fresh_snow('fsd.tif', (700097.9845, 3581763.7627))
