import gdal
import numpy as np
import affine
import os
import glob

EFF_AIR = 1.0005
EFF_ICE = 3.179
ICE_DENSITY = 0.917  # gm/cc
FRESH_SNOW_DENSITY = 0.06  # gm/cc
WAVELENGTH = 3.10880853  # cm
NO_DATA_VALUE = -32768
MEAN_INC_ANGLE = (38.072940826416016 + 39.38078689575195 + 38.10858917236328 + 39.38400650024414)/4.
DHUNDI_COORDS = (700089.771, 3581794.5556)  # UTM 43N


def read_images(path, imgformat='*.tif'):
    """
    Read images in a directory
    :param path: Directory path
    :param imgformat: Type of image file
    :return: Dictionary of GDAL Opened references/ pointers to specific files
    """

    print("Reading images...")
    images = {}
    files = os.path.join(path, imgformat)
    for file in glob.glob(files):
        key = file[file.rfind('/') + 1: file.rfind('.')]
        images[key] = gdal.Open(file)
    print("Finished reading")
    return images


def get_complex_image(img_file, is_dual=False):
    """
    Read complex image stored either in four bands or two separate bands and set nan values accordingly
    :param img_file: GDAL reference file
    :param is_dual: True if image file is stored in two separate bands
    :return: If dual is set true, a complex numpy array is returned, numpy array tuple otherwise
    """

    mst = img_file.GetRasterBand(1).ReadAsArray() + img_file.GetRasterBand(2).ReadAsArray() * 1j
    mst[mst == np.complex(NO_DATA_VALUE, NO_DATA_VALUE)] = np.nan
    if not is_dual:
        slv = img_file.GetRasterBand(3).ReadAsArray() + img_file.GetRasterBand(4).ReadAsArray() * 1j
        slv[slv == np.complex(NO_DATA_VALUE, NO_DATA_VALUE)] = np.nan
        return mst, slv
    return mst


def get_image_array(img_file):
    """
    Read real numpy arrays from file
    :param img_file: GDAL reference file
    :return: Numpy array with nan set accordingly
    """

    arr = img_file.GetRasterBand(1).ReadAsArray()
    arr[arr == NO_DATA_VALUE] = np.nan
    return arr


def set_nan_img(img_arr, layover_file, forest_file):
    """
    Set nan values to specific images using layover and forest masks
    :param img_arr: Image array whose nan values are to be set
    :param layover_file: Layover file, no layover marked with 0
    :param forest_file: Forest file, forest marked with 0
    :return: Nan set array
    """

    layover = get_image_array(layover_file)
    forest = get_image_array(forest_file)
    for idx, lval in np.ndenumerate(layover):
        if np.round(lval) != 0 or forest[idx] == 0:
            img_arr[idx] = np.nan
    return img_arr


def nanfix_tmat_val(tmat, idx, verbose=True):
    """
    Fix nan value occuring due to incorrect terrain correction
    :param tmat: Complex coherency matrix
    :param idx: Index at which the nan value is to be replaced by the mean of its neighbourhood
    :param verbose: Set true for detailed logs
    :return: Corrected element
    """

    i = 1
    while True:
        window = get_ensemble_window(tmat, idx, (i, i))
        tval = np.nanmean(window)
        if verbose:
            print('\nTVAL nanfix', i, np.abs(tval))
        if not np.isnan(tval):
            return tval
        i += 1


def nanfix_tmat_arr(tmat_arr, lia_arr, layover_arr=None, forest_arr=None, apply_masks=True, verbose=False):
    """
    Fix nan values occuring due to incorrect terrain correction
    :param tmat_arr: Complex coherency matrix
    :param lia_arr: Local incidence angle array or any coregistered image array having non-nan values
    in the area of interest
    :param layover_arr: Layover array, no layover marked with 0
    :param forest_arr: Forest array, forest marked with 0
    :param apply_masks: Set true for applying layover and forest masks
    :param verbose: Set true for detailed logs
    :return:
    """

    for idx, tval in np.ndenumerate(tmat_arr):
        if not np.isnan(lia_arr[idx]):
            if np.isnan(tval):
                if apply_masks:
                    if np.round(layover_arr[idx]) == 0 and forest_arr[idx] == 1:
                        tmat_arr[idx] = nanfix_tmat_val(tmat_arr, idx, verbose)
                else:
                    tmat_arr[idx] = nanfix_tmat_val(tmat_arr, idx, verbose)
    return tmat_arr


def write_file(arr, src_file, outfile='test', no_data_value=NO_DATA_VALUE, is_complex=True):
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
    if is_complex:
        out = driver.Create(outfile + ".tif", arr.shape[1], arr.shape[0], 2, gdal.GDT_Float32)
    else:
        out = driver.Create(outfile + ".tif", arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    out.SetProjection(src_file.GetProjection())
    out.SetGeoTransform(src_file.GetGeoTransform())
    out.GetRasterBand(1).SetNoDataValue(no_data_value)
    if is_complex:
        arr[np.isnan(arr)] = no_data_value + no_data_value * 1j
        out.GetRasterBand(2).SetNoDataValue(no_data_value)
        out.GetRasterBand(1).WriteArray(arr.real)
        out.GetRasterBand(2).WriteArray(arr.imag)
    else:
        arr[np.isnan(arr)] = no_data_value
        out.GetRasterBand(1).WriteArray(arr)
    out.FlushCache()


def get_depolarisation_factor(axial_ratio, shape):
    """
    Calculation three depolarisation factors
    :param axial_ratio: Axial ratio
    :param shape: Particle shape to consider
    :return: Tuple containing three depolarisation factors in x, y, and z directions
    """

    depolarisation_factorx = depolarisation_factory = depolarisation_factorz = 1/3.
    if shape == 'o':
        eccentricity = np.sqrt(axial_ratio ** 2 - 1)
        depolarisation_factorz = (1 + eccentricity**2) * (eccentricity - np.arctan(eccentricity)) / eccentricity ** 3
        depolarisation_factorx = depolarisation_factory = 0.5 * (1 - depolarisation_factorz)
    elif shape == 'p':
        eccentricity = np.sqrt(1 - axial_ratio ** 2)
        depolarisation_factorx = ((1 - eccentricity ** 2) *
                                  (np.log((1 + eccentricity) / (1 - eccentricity))
                                   - 2 * eccentricity)) / (2 * eccentricity ** 3)
        depolarisation_factory = depolarisation_factorz = 0.5 * (1 - depolarisation_factorx)
    return depolarisation_factorx, depolarisation_factory, depolarisation_factorz


def get_effective_permittivity(fvol, depolarisation_factor):
    """
    Calculate effective permittivity
    :param fvol: Snow volume fraction
    :param depolarisation_factor: Depolarisation factor
    :return: Effective permittivity
    """

    eff_diff = EFF_ICE - EFF_AIR
    eff = EFF_AIR * (1 + fvol * eff_diff/(EFF_AIR + (1 - fvol) * depolarisation_factor * eff_diff))
    return eff


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


def check_values(img_arr, img_file, geocoords, nsize=(1, 1), is_complex=False, full_stat=False):
    """
    Validate results
    :param img_arr: Image array to validate
    :param img_file: Original GDAL reference having affine transformation parameters
    :param geocoords: Geo-coordinates in tuple format
    :param nsize: Validation window size (should be half of the desired window size)
    :param is_complex: Set true for complex images such as the coherency image
    :param full_stat: Return min, max, mean and standard deviation if true, mean and sd if false
    :return: Tuple containing statistics
    """

    px, py = retrieve_pixel_coords(geocoords, img_file)
    if is_complex:
        img_arr = np.abs(img_arr)
    img_loc = get_ensemble_window(img_arr, (py, px), nsize)
    mean = np.nanmean(img_loc)
    sd = np.nanstd(img_loc)
    if full_stat:
        return np.nanmin(img_loc), np.nanmax(img_loc), mean, sd
    return mean, sd


def get_ensemble_window(image_arr, index, wsize):
    """
    Subset image array based on the window size
    :param image_arr: Image array whose subset is to be returned
    :param index: Central subset index
    :param wsize: Ensemble window size (should be half of desired window size)
    :return: Subset array
    """

    startx = index[0] - wsize[0]
    starty = index[1] - wsize[1]
    if startx < 0:
        startx = 0
    if starty < 0:
        starty = 0
    endx = index[0] + wsize[0] + 1
    endy = index[1] + wsize[1] + 1
    limits = image_arr.shape[0] + 1, image_arr.shape[1] + 1
    if endx > limits[0] + 1:
        endx = limits[0] + 1
    if endy > limits[1] + 1:
        endy = limits[1] + 1
    return image_arr[startx: endx, starty: endy]


def get_ensemble_avg(image_arr, wsize, image_file, outfile, stat='mean', verbose=True, wf=False, is_complex=False):
    """
    Perform Ensemble Filtering based on mean, median or maximum
    :param image_arr: Image array to filter
    :param wsize: Ensemble window size (should be half of desired window size)
    :param image_file: Original GDAL reference for writing output image
    :param outfile: Outfile file path
    :param stat: Statistics to use while ensemble filtering (mean, med, max)
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :param is_complex: Set true for averaging complex numbers
    :return: Ensemble filtered array
    """

    dt = np.float32
    if is_complex:
        dt = np.complex
    emat = np.full_like(image_arr, np.nan, dtype=dt)
    for index, value in np.ndenumerate(image_arr):
        if not np.isnan(value):
            ensemble_window = get_ensemble_window(image_arr, index, wsize)
            if stat == 'mean':
                emat[index] = np.nanmean(ensemble_window)
            elif stat == 'med':
                emat[index] = np.nanmedian(ensemble_window)
            elif stat == 'max':
                emat[index] = np.nanmax(ensemble_window)
            if verbose:
                print(index, emat[index])
    if wf:
        outfile = 'Out/' + outfile
        np.save(outfile, emat)
        write_file(emat.copy(), image_file, outfile, is_complex=is_complex)
    return emat


def calc_coh_mat(s_hh, s_vv, img_dict, outfile, num_looks=10, apply_masks=True, verbose=True, wf=True):
    """
    Calculate complex coherency matrix based on looks
    :param s_hh: HH array
    :param s_vv: VV array
    :param img_dict: Image dictionary containing GDAL references
    :param outfile: Output file path
    :param num_looks: Number of looks to apply
    :param apply_masks: Set true for applying layover and forest masks
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :return: Nan fixed complex coherency matrix
    """

    tmat = np.full_like(s_hh, np.nan, dtype=np.complex)
    max_y = tmat.shape[1]
    for itr in np.ndenumerate(tmat):
        idx = itr[0]
        start_x = idx[0]
        start_y = idx[1]
        end_y = start_y + num_looks
        if end_y > max_y:
            end_y = max_y
        sub_hh = s_hh[start_x][start_y: end_y]
        sub_vv = s_vv[start_x][start_y: end_y]
        sub_num = sub_vv * np.conj(sub_hh)
        nan_check = np.isnan(np.array([[sub_hh[0], sub_vv[0], sub_num[0]]]))
        if len(nan_check[nan_check]) == 0:
            num = np.nansum(sub_num)
            denom = np.sqrt(np.nansum(sub_vv * np.conj(sub_vv))) * np.sqrt(np.nansum(sub_hh * np.conj(sub_hh)))
            tmat[idx] = num / denom
            if np.abs(tmat[idx]) > 1:
                tmat[idx] = 1 + 0j
            if verbose:
                print('Coherence at ', idx, '= ', np.abs(tmat[idx]))
    lia_arr = get_image_array(img_dict['LIA'])
    if apply_masks:
        layover_arr = get_image_array(img_dict['LAYOVER'])
        forest_arr = get_image_array(img_dict['FOREST'])
        tmat = nanfix_tmat_arr(tmat, lia_arr, layover_arr, forest_arr)
    else:
        tmat = nanfix_tmat_arr(tmat, lia_arr, apply_masks=False)
    if wf:
        np.save('Out/Coherence_' + outfile, tmat)
        write_file(tmat.copy(), img_dict['LIA'], 'Out/Coherence_' + outfile)
    return tmat


def calc_ensemble_cohmat(s_hh, s_vv, img_dict, outfile, wsize=(5, 5), apply_masks=True, verbose=True, wf=False):
    """
    Calculate complex coherency matrix based on ensemble averaging
    :param s_hh: HH array
    :param s_vv: VV array
    :param img_dict: Image dictionary containing GDAL references
    :param outfile: Output file path
    :param wsize: Ensemble window size (should be half of desired window size)
    :param apply_masks: Set true for applying layover and forest masks
    :param verbose: Set true for detailed logs
    :param wf: Set true to save intermediate results
    :return: Nan fixed complex coherency matrix
    """

    lia_file = img_dict['LIA']
    num = get_ensemble_avg(s_vv * np.conj(s_hh), wsize=wsize, image_file=lia_file, outfile='Num', is_complex=True,
                           verbose=verbose, wf=False)
    d1 = get_ensemble_avg((s_vv * np.conj(s_vv)).real, wsize=wsize, image_file=lia_file, outfile='D1',
                          verbose=verbose, wf=False)
    d2 = get_ensemble_avg((s_hh * np.conj(s_hh)).real, wsize=wsize, image_file=lia_file, outfile='D2',
                          verbose=verbose, wf=False)

    tmat = num / (np.sqrt(d1 * d2))
    tmat[np.abs(tmat) > 1] = 1 + 0j

    lia_arr = get_image_array(lia_file)
    if apply_masks:
        layover_arr = get_image_array(img_dict['LAYOVER'])
        forest_arr = get_image_array(img_dict['FOREST'])
        tmat = nanfix_tmat_arr(tmat, lia_arr, layover_arr, forest_arr)
    else:
        tmat = nanfix_tmat_arr(tmat, lia_arr, apply_masks=False)
    if wf:
        np.save('Out/Coherence_Ensemble_' + outfile, tmat)
        write_file(tmat.copy(), lia_file, 'Out/Coherence_Ensemble_' + outfile)
    return tmat


def calc_cpd(image_dict, wsize=(2, 2), apply_masks=True, verbose=False, wf=True, load_files=False, from_coh=False,
             coh_type='E'):
    """
    Calculate copolar phase difference from complex coherency matrix
    :param image_dict: Image dictionary containing GDAL references
    :param wsize: Ensemble window size (should be half of the desired window)
    :param apply_masks: Set true to apply layover and forest masks
    :param verbose: Set true for log details
    :param wf: Set true to save intermediate files
    :param load_files: Set true to load existing numpy binary files and skip computation
    :param from_coh: Set true to calculate CPD from complex coherence
    :param coh_type: Type of coherence averaging, 'L' for look based, 'E' for ensemble based
    :return: Tuple containing averaged CPD array and real coherence array
    """

    if not load_files:
        hh_file = image_dict['HH']
        vv_file = image_dict['VV']
        lia_file = image_dict['LIA']
        layover_file = image_dict['LAYOVER']
        forest_file = image_dict['FOREST']

        hh_mst, hh_slv = get_complex_image(hh_file)
        vv_mst, vv_slv = get_complex_image(vv_file)

        hh_mst = set_nan_img(hh_mst, layover_file, forest_file)
        hh_slv = set_nan_img(hh_slv, layover_file, forest_file)
        vv_mst = set_nan_img(vv_mst, layover_file, forest_file)
        vv_slv = set_nan_img(vv_slv, layover_file, forest_file)

        if from_coh:
            if coh_type == 'E':
                coh_mat_mst = calc_ensemble_cohmat(hh_mst, vv_mst, image_dict, wsize=wsize, apply_masks=apply_masks,
                                                   verbose=verbose, outfile='HH', wf=False)
                coh_mat_slv = calc_ensemble_cohmat(hh_slv, vv_slv, image_dict, wsize=wsize, apply_masks=apply_masks,
                                                   verbose=verbose, outfile='VV', wf=False)
            else:
                ws = wsize[0] * 2 + 1
                coh_mat_mst = calc_coh_mat(hh_mst, vv_mst, image_dict, num_looks=ws, apply_masks=apply_masks,
                                           verbose=verbose, outfile='HH', wf=False)
                coh_mat_slv = calc_coh_mat(hh_slv, vv_slv, image_dict, num_looks=ws, apply_masks=apply_masks,
                                           verbose=verbose, outfile='VV', wf=False)
            cpd_mst = np.arctan2(coh_mat_mst.imag, coh_mat_mst.real)
            cpd_slv = np.arctan2(coh_mat_slv.imag, coh_mat_slv.real)
            cpd_avg = (cpd_mst + cpd_slv) / 2.
            coh_avg = np.abs((coh_mat_mst + coh_mat_slv) / 2)
        else:
            cpd_mst = np.arctan2(vv_mst.imag, vv_mst.real) - np.arctan2(hh_mst.imag, hh_mst.real)
            cpd_slv = np.arctan2(vv_slv.imag, vv_slv.real) - np.arctan2(hh_slv.imag, hh_slv.real)
            cpd_avg = (cpd_mst + cpd_slv) / 2.
            cpd_avg = get_ensemble_avg(cpd_avg, wsize=wsize, image_file=lia_file, outfile='CPD', verbose=verbose,
                                       wf=False, is_complex=False)
        lia_arr = get_image_array(lia_file)
        if apply_masks:
            layover_arr = get_image_array(layover_file)
            forest_arr = get_image_array(forest_file)
            cpd_avg = nanfix_tmat_arr(cpd_avg, lia_arr, layover_arr, forest_arr)
        else:
            cpd_avg = nanfix_tmat_arr(cpd_avg, lia_arr, apply_masks=False)
        if not from_coh:
            coh_avg = cpd_avg.copy()
            coh_avg[~np.isnan(coh_avg)] = 1
        if wf:
            np.save('Out/CPD_Avg', cpd_avg)
            np.save('Out/Coh_Avg', coh_avg)
    else:
        cpd_avg = np.load('Out/CPD_Avg.npy')
        coh_avg = np.load('Out/Coh_Avg.npy')
    return cpd_avg, coh_avg


def cpd2freshsnow(cpd_arr, lia_file, coh_arr, ssd_file, coh_threshold, axial_ratio=2, shape='o', verbose=True,
                  wf=True, load_file=False, fsd_threshold=100):
    """
    Compute fresh snow depth from CPD
    :param cpd_arr: CPD array
    :param lia_file: Local incidence angle GDAL reference
    :param coh_arr: Real valued coherence array
    :param ssd_file: Standing snow depth GDAL reference, SSD > FSD condition must hold true
    :param coh_threshold: Coherence threshold
    :param axial_ratio: Axial ratio
    :param shape: Shape of snow particle
    :param verbose: Set true for log details
    :param wf: Set true to write intermediate files
    :param load_file: Set true to load existing FSD numpy binary and skip computation
    :param fsd_threshold: Maximum possible fresh snow depth (cm) in the study area, outlier values are set to zero
    :return: Fresh snow depth array
    """

    if not load_file:
        fvol = FRESH_SNOW_DENSITY/ICE_DENSITY
        depolarisation_factors = get_depolarisation_factor(axial_ratio, shape)
        eff_h = get_effective_permittivity(fvol, depolarisation_factors[0])
        eff_y = get_effective_permittivity(fvol, depolarisation_factors[1])
        eff_z = get_effective_permittivity(fvol, depolarisation_factors[2])

        lia_arr = np.deg2rad(get_image_array(lia_file))
        fsd_arr = np.full_like(cpd_arr, np.nan, dtype=np.float32)
        ssd_arr = get_image_array(ssd_file)
        for index, cpd in np.ndenumerate(cpd_arr):
            if not np.isnan(cpd):
                fsd_val = 0
                if cpd > 0 and coh_arr[index] > coh_threshold:
                    lia_val = lia_arr[index]
                    sin_inc_sq = np.sin(lia_val) ** 2
                    eff_v = eff_y * np.cos(lia_val) ** 2 + eff_z * sin_inc_sq
                    xeta_diff = np.sqrt(eff_v - sin_inc_sq) - np.sqrt(eff_h - sin_inc_sq)
                    if xeta_diff < 0:
                        fsd_val = np.float32(-cpd * WAVELENGTH / (4 * np.pi * xeta_diff))
                        if fsd_val >= fsd_threshold or fsd_val >= ssd_arr[index]:
                            fsd_val = 0
                fsd_arr[index] = fsd_val
                if verbose:
                    print('FSD=', index, fsd_val)
        if wf:
            np.save('Out/FSD_Unf', fsd_arr)
    else:
        fsd_arr = np.load('Out/FSD_Unf.npy')
    return fsd_arr


def get_fresh_swe(fsd_arr, density, img_file, wf=True):
    """
    Calculate fresh snow water equivalent (SWE) in mm or kg/m^3
    :param fsd_arr: Fresh snow depth array in cm
    :param density: Snow density (scalar or array) in g/cm^3
    :param img_file: Original image file containing affine transformation parameters
    :param wf: Set true to write intermediate files
    :return: SWE array
    """

    swe = fsd_arr * density * 10
    if wf:
        np.save('Out/FSD_SWE', swe)
        write_file(swe.copy(), img_file, outfile='Out/FSD_SWE', is_complex=False)
    return swe


def get_wishart_class_stats(input_wishart, layover_file):
    """
    Calculate Wishart class percentages
    :param input_wishart: Wishart classified image path
    :param layover_file: Layover file path
    :return: None
    """

    print('File: ', input_wishart)
    input_wishart = gdal.Open(input_wishart)
    layover_file = gdal.Open(layover_file)
    wishart_arr = input_wishart.GetRasterBand(1).ReadAsArray()
    layover_arr = layover_file.GetRasterBand(1).ReadAsArray()
    new_arr = np.full_like(wishart_arr, NO_DATA_VALUE, dtype=np.int32)
    print('Checking valid pixels...')
    for index, value in np.ndenumerate(wishart_arr):
        if value != 0 and layover_arr[index] == 0:
            new_arr[index] = int(round(value))
    classes, count = np.unique(new_arr, return_counts=True)
    total_pixels = np.sum(count)
    print('Total pixels=', total_pixels)
    class_percent = np.float32(count * 100. / total_pixels)
    print(classes, class_percent)


def sensitivity_analysis(image_dict):
    """
    Main caller function for FSD sensitivity analysis
    :param image_dict: Image dictionary containing GDAL references
    :return: None
    """

    # wrange = range(3, 66, 2)
    # fwindows = [(i, j) for i, j in zip(wrange, wrange)]
    # cwindows = fwindows.copy()
    fwindows = [(49, 49)]
    cwindows = [(3, 3)]
    coh_threshold = [0]
    apply_masks = True
    verbose = False
    wf = True
    lf = False
    lia_file = image_dict['LIA']
    outfile = open('sensitivity_fsd_swe.csv', 'a+')
    outfile.write('CWindow CThreshold FWindow Mean_FSD(cm) SD_FSD(cm) Mean_SWE(mm) SD_SWE(mm)\n')
    print('Computation started...')
    for wsize in cwindows:
        ws1, ws2 = int(wsize[0] / 2.), int(wsize[1] / 2.)
        wstr1 = '(' + str(wsize[0]) + ',' + str(wsize[1]) + ')'
        print('Computing CPD and Coherence...')
        cpd_arr, coh_arr = calc_cpd(image_dict, (ws1, ws2), apply_masks=apply_masks, verbose=verbose, wf=wf,
                                    load_files=lf, from_coh=False, coh_type='L')
        for ct in coh_threshold:
                print('Calculating fresh snow depth ...')
                fsd_arr = cpd2freshsnow(cpd_arr, lia_file, coh_arr, ssd_file=image_dict['SSD'], coh_threshold=ct,
                                        verbose=verbose, wf=wf, load_file=lf, fsd_threshold=200)
                for fsize in fwindows:
                    fs1, fs2 = int(fsize[0] / 2.), int(fsize[1] / 2.)
                    print('FSD Ensemble Averaging')
                    fsd_avg = get_ensemble_avg(fsd_arr, (fs1, fs2), lia_file, 'FSD_49_C3', verbose=verbose, wf=wf)
                    swe = get_fresh_swe(fsd_avg, FRESH_SNOW_DENSITY, img_file=lia_file)
                    vr = check_values(fsd_avg, lia_file, DHUNDI_COORDS)
                    vr_str1 = ' '.join([str(r) for r in vr])
                    wstr2 = '(' + str(fsize[0]) + ',' + str(fsize[1]) + ')'
                    vr = check_values(swe, lia_file, DHUNDI_COORDS)
                    vr_str2 = ' '.join([str(r) for r in vr])
                    final_str = wstr1 + ' ' + str(ct) + ' ' + wstr2 + ' ' + vr_str1 + ' ' + vr_str2 + '\n'
                    print(final_str)
                    outfile.write(final_str)


image_dict = read_images('../../THESIS/Thesis_Files/Polinsar/Clipped_Tifs')
sensitivity_analysis(image_dict)

