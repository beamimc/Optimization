# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:01:34 2021

@author: es0055
"""

from osgeo import ogr
import pandas as pd



sol = pd.read_csv('sol_optima.csv')
points = pd.read_csv('Coordenadas_Href_Mundi.csv')
# # you have a list of points
# listPoint = [[13.415449261665342, 52.502674590782519],[13.416039347648621, 52.50250152147968],[13.415787220001221, 52.501845158120446],[13.416162729263306, 52.502201097675766],[13.415406346321104, 52.502334982450677],[13.415111303329468,52.50204435400651]]
# # Add the points to the ring
# ring = ogr.Geometry(ogr.wkbLinearRing)
# for point in listPoint:
#     lat = point[0]
#     lon = point[1]
#     print(lat, lon)
#     ring.AddPoint(lat,lon)


# # Add first point again to ring to close polygon
# ring.AddPoint(listPoint[0][0], listPoint[0][1])

# # Add the ring to the polygon
# poly = ogr.Geometry(ogr.wkbPolygon)
# poly.AddGeometry(ring)
# print(poly.ExportToJson())