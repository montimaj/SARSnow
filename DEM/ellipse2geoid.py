import numpy as np
from scipy.interpolate import RectBivariateSpline as Spline
import pygeodesy as geo
from pygeodesy.ellipsoidalVincenty import LatLon


class EGM96():  # WGS 84 Ellipsoid
    # http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/binarygeoid.html
    # Download WW15MGH.DAC
    def __init__(self, binFile):
        egm = np.fromfile(binFile, '>i2').reshape(721, 1440) / 100
        longs = np.arange(0, 360, 0.25)
        lats = np.arange(-90, 90.1, 0.25)
        self.interp = Spline(lats, longs, egm)

    def height(self, longitude, latitude):
        latitude *= -1
        # longitude[longitude < 0] += 360
        if longitude < 0:
            longitude += 360
        return self.interp.ev(latitude, longitude)

# nad = geo.datum.Datums.NAD83

# wgsCoord = LatLon( lat, lon, elev - Z, wgs)
# nadCoord = wgsCoord.convertDatum(nad)
# navd = Geoid12B('g2012bu5.bin')
# navd88Elev = nadCoord.height + navd.height(nadCoord.lon, nadCoord.lat)