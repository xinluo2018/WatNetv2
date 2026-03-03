## author: luo xin, 
## creat: 2021.6.15, modify: 2025.6.17
## des: coordinates transform for point and raster pixel.

import numpy as np
from pyproj import Transformer

def coor2coor(crs_from, crs_to, x, y):
    """
    Transform coordinates from crs_from to crs_to
    input:
        crs_from and crs_to are EPSG number (e.g., 4326, 3031)
        x and y are x-coord and y-coord corresponding to crs_from and crs_to    
    return:
        x-coord and y-coord in crs_to 
    """
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    x_new, y_new = transformer.transform(x, y)
    return x_new, y_new

def geo2imagexy(x, y, gdal_trans):
    '''
    des: from georeferenced location (i.e., lon, lat) to image location(col,row).
    input:
        gdal_proj: obtained by gdal.Open() and .GetGeoTransform(), or by geotif_io.readTiff()['geotrans']
        x: project or georeferenced x, i.e.,lon
        y: project or georeferenced y, i.e., lat
    return: 
        image col and row corresponding to the georeferenced location.
    '''
    a = np.array([[gdal_trans[1], gdal_trans[2]], [gdal_trans[4], gdal_trans[5]]])
    b = np.array([x - gdal_trans[0], y - gdal_trans[3]])
    col_img, row_img = np.linalg.solve(a, b)
    col_img, row_img = np.floor(col_img).astype('int'), np.floor(row_img).astype('int')
    return row_img, col_img

def imagexy2geo(row, col, gdal_trans):
    '''
    input: 
        img_gdal: GDAL data (read by gdal.Open()
        row and col are corresponding to input image (dataset)
    :return:  
        geographical coordinates (left up of pixel)
    '''
    x = gdal_trans[0] + col * gdal_trans[1] + row * gdal_trans[2]
    y = gdal_trans[3] + col * gdal_trans[4] + row * gdal_trans[5]
    return x, y



