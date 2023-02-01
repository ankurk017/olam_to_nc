import xarray as X
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import matplotlib.ticker as mticker
from scipy.interpolate import griddata
import glob


#====================================
# FUNCTION TO ADD THE COASTLINES IN THE PLOT
def add_coast(axes):

	countries = cfeature.NaturalEarthFeature(scale='50m',category='cultural',name='admin_0_countries',facecolor='none') #add countries
	states = cfeature.NaturalEarthFeature(scale='50m',category='cultural', name='admin_1_states_provinces_lines',facecolor='none') # add states/provinces
	axes.add_feature(countries,edgecolor='w',linewidth=2) # plotting county borders
	axes.add_feature(states,edgecolor='w',linewidth=2) # plotting state/province borders
	
	gl = axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,  linewidth=2,  color="gray", alpha=0, linestyle="--" )
	gl.xlabels_top = False
	gl.ylabels_left = True
	gl.ylabels_right= False
	gl.xlines = True
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	gl.ylabels_left = True
	gl.ylabels_right= False
	gl.xlines = True
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER

	return None

A= X.open_dataset('test.nc')

x=A['lon']
y=A['lat']
extent = np.min(x), np.max(x), np.min(y), np.max(y)


fig = plt.figure(figsize=(9,7))
axes = plt.axes(projection=ccrs.PlateCarree())

img = axes.imshow(A['LAND%VEG_NDVIC'], extent=extent,transform=ccrs.PlateCarree(),cmap='jet')
cbar = plt.colorbar(img,orientation='horizontal',pad=.07)
add_coast(axes)

plt.show()


