import xarray as X
from scipy import interpolate
import numpy as np
from scipy.interpolate import griddata
import datetime as dt
import glob
import pandas as pd

#====================================

def interpolate_grids(olam_out, latlons,  outgrids, var_list, variables, mask=None):
	for index in var_list:
		values=olam_out[index]
		print(values.shape, index)
		values = values[1:]
		interp_data=np.squeeze(griddata(latlons, values, outgrids, method='nearest'))
		variables[index]=(['lat','lon'],interp_data)
		if mask is not None:
			variables[index]=(['lat','lon'],interp_data*mask*1)

	return variables


def interpolate_3grids(olam_out, latlons,  outgrids, var_list, variables, mask=None):
	for index in var_list:
		values=olam_out[index]
		print(values.shape, index)
		values = values[1:,:]
		interp_data=[np.squeeze(griddata(latlons, values[:,levels], outgrids, method='nearest')) for levels in range(values.shape[1])]
		variables[index]=(['level','lat','lon'],np.array(interp_data))
		if mask is not None:
			variables[index]=(['level','lat','lon'],np.array(interp_data)*mask*1)
	return variables


def variable_list():
	
	var_list1=["ACCPA","ACCPD","ACCPG","ACCPH","ACCPP","ACCPR","ACCPS","ACONPR","ALBEDT","CBMF","CONPRR","COSZ","DPCPGR","KCUBOT","KCUTOP","KPBLH","PBLH","PCPGR","PCPRA","PCPRD","PCPRG","PCPRH","PCPRP","PCPRR","PCPRS","QPCPGR","RLONG","RLONGUP","RLONGUP_ACCUM","RLONGUP_CLR","RLONGUP_CLR_ACCUM","RLONGUP_TOP","RLONGUP_TOP_ACCUM","RLONGUP_TOP_CLR","RLONGUP_TOP_CLR_ACCUM","RLONG_ACCUM","RLONG_CLR","RLONG_CLR_ACCUM","RSHORT","RSHORTUP_ACCUM","RSHORTUP_CLR","RSHORTUP_CLR_ACCUM","RSHORTUP_TOP","RSHORTUP_TOP_ACCUM","RSHORTUP_TOP_CLR","RSHORTUP_TOP_CLR_ACCUM","RSHORT_ACCUM","RSHORT_CLR","RSHORT_CLR_ACCUM","RSHORT_TOP","RSHORT_TOP_ACCUM","RSHORT_TOP_CLR","RSHORT_TOP_CLR_ACCUM","SFLUXR","SFLUXR_ACCUM","SFLUXT","SFLUXT_ACCUM","USTAR","WSTAR"]


	var_list2A=["LAND%ALBEDO_BEAM","LAND%ALBEDO_DIFFUSE","LAND%CANSHV","LAND%CANTEMP","LAND%CAN_DEPTH","LAND%COSZ","LAND%DPCPG","LAND%GGAER","LAND%GROUND_SHV","LAND%HCAPVEG","LAND%HEAD0","LAND%HEAD1","LAND%NLEV_SFCWATER","LAND%PCPG","LAND%QPCPG","LAND%RIB","LAND%RLONG","LAND%RLONGUP","LAND%RLONG_ALBEDO","LAND%RLONG_G","LAND%RLONG_S","LAND%RLONG_V","LAND%ROUGH","LAND%RSHORT","LAND%RSHORT_DIFFUSE","LAND%RSHORT_G","LAND%RSHORT_S","LAND%RSHORT_V","LAND%SFCWATER_DEPTH","LAND%SFCWATER_ENERGY","LAND%SFCWATER_MASS","LAND%SFLUXC","LAND%SFLUXR","LAND%SFLUXT","LAND%SNOWFAC","LAND%STOM_RESIST","LAND%SURFACE_SSH","LAND%SXFER_C","LAND%SXFER_R","LAND%SXFER_T","LAND%USTAR","LAND%VEG_ALBEDO","LAND%VEG_FRACAREA","LAND%VEG_HEIGHT","LAND%VEG_LAI","LAND%VEG_NDVIC","LAND%VEG_ROUGH","LAND%VEG_TAI","LAND%VEG_TEMP","LAND%VEG_WATER","LAND%VF","LAND%VKMSFC"]
	#%SOIL_ENERGY","LAND%SOIL_WATER", not included in this list
	var_list2B=["AIRSHV_L_ACCUM","AIRTEMP_L_ACCUM","CANSHV_L_ACCUM","CANTEMP_L_ACCUM","SFLUXR_L_ACCUM","SFLUXT_L_ACCUM","SKINTEMP_L_ACCUM","VELS_L_ACCUM","WXFER1_L_ACCUM"]
	# MERGING VARIABLES WITH THE SAME SHAPE
	var_list2 = var_list2A + var_list2B



	var_list3A=["SEA%ALBEDO_BEAM","SEA%ALBEDO_DIFFUSE","SEA%CANSHV","SEA%CANTEMP","SEA%CAN_DEPTH","SEA%DPCPG","SEA%ICE_ALBEDO","SEA%ICE_CANSHV","SEA%ICE_CANTEMP","SEA%ICE_NET_RLONG","SEA%ICE_NET_RSHORT","SEA%ICE_RLONGUP","SEA%ICE_ROUGH","SEA%ICE_SFC_SSH","SEA%ICE_SFLUXR","SEA%ICE_SFLUXT","SEA%ICE_SXFER_R","SEA%ICE_SXFER_T","SEA%ICE_USTAR","SEA%ICE_VKMSFC","SEA%ICE_WTHV","SEA%NLEV_SEAICE","SEA%PCPG","SEA%QPCPG","SEA%RLONG","SEA%RLONGUP","SEA%RLONG_ALBEDO","SEA%ROUGH","SEA%RSHORT","SEA%RSHORT_DIFFUSE","SEA%SEAICEC","SEA%SEATC","SEA%SEA_ALBEDO","SEA%SEA_CANSHV","SEA%SEA_CANTEMP","SEA%SEA_RLONGUP","SEA%SEA_ROUGH","SEA%SEA_SFC_SSH","SEA%SEA_SFLUXR","SEA%SEA_SFLUXT","SEA%SEA_SXFER_R","SEA%SEA_SXFER_T","SEA%SEA_USTAR","SEA%SEA_VKMSFC","SEA%SEA_WTHV","SEA%SFLUXR","SEA%SFLUXT","SEA%SURFACE_SSH","SEA%SXFER_R","SEA%SXFER_T","SEA%USTAR","SEA%VKMSFC"]
	#SEA%SEAICE_ENERGY, SEA%SEAICE_TEMPK, not included in this list
	var_list3B=["AIRSHV_S_ACCUM","AIRTEMP_S_ACCUM","CANSHV_S_ACCUM","CANTEMP_S_ACCUM","SFLUXR_S_ACCUM","SFLUXT_S_ACCUM","SKINTEMP_S_ACCUM","VELS_S_ACCUM"]
	# MERGING VARIABLES WITH THE SAME SHAPE
	var_list3 = var_list3A+var_list3B

	var_list4=["CLOUD_FRAC","CON_P","FQTPBL","FTHPBL","FTHRD_LW","FTHRD_SW","HKH","HKM","PRESS","PRESS_ACCUM","Q2","Q6","Q7","QWCON","RHO","RTSRC","SH_A","SH_C","SH_D","SH_G","SH_H","SH_P","SH_R","SH_S","SH_V","SH_V_ACCUM","SH_W","TAIR","TAIR_ACCUM","THETA","THIL","THSRC","VXSRC","VYSRC","VZSRC","WC","WC_ACCUM","WMC"]

	return var_list1, var_list2, var_list3, var_list4

#====================================

gfile='/nas/rstor/akumar/OLAM/olam-4.12.2/build_test/sfcfiles/gridfile2.h5'
lfile='/nas/rstor/akumar/OLAM/olam-4.12.2/build_test/sfcfiles/landh2.h5'
sfile='/nas/rstor/akumar/OLAM/olam-4.12.2/build_test/sfcfiles/seah2.h5'
olamout = 'Bahamas-H-2019-10-04-163000.h5'


gfile='/rstor/akumar/USA/Hurricane_IDA/OLAM/olam-4.12.2/build_test/sfcfiles/gridfile2.h5'
lfile='/rstor/akumar/USA/Hurricane_IDA/OLAM/olam-4.12.2/build_test/sfcfiles/landh2.h5'
sfile='/rstor/akumar/USA/Hurricane_IDA/OLAM/olam-4.12.2/build_test/sfcfiles/seah2.h5'
olamout = '/rstor/akumar/USA/Hurricane_IDA/OLAM/olam-4.12.2/build_test/hist/IDA_20210829-H-2021-08-29-120000.h5'

gfile='/rtmp/gpriftis/olam_test_files/gridfile2.h5'
lfile='/rtmp/gpriftis/olam_test_files/landh2.h5'
sfile='/rtmp/gpriftis/olam_test_files/seah2.h5'
olamout = '/rtmp/gpriftis/olam_test_files/LB_Brunei-H-2009-08-06-030000.h5'

gridfile=X.open_dataset(gfile)
landfile=X.open_dataset(lfile)
seafile=X.open_dataset(sfile)


#====================================
# MENTION THE BOUNDING BOX OF THE PLOT

lon=np.arange(-79.5,-77.7,0.02)
lat=np.arange(26,27.2,0.02)

lon=np.arange(-100,-80,0.1)
lat=np.arange(20,40,0.1)



lon=np.arange(100, 120 ,0.1)
lat=np.arange(-5,16,0.1)

grid_x, grid_y=np.meshgrid(lon, lat)
outgrids=(grid_x, grid_y)
#====================================
olam_out=X.open_dataset(olamout)
# FOR READING THE TIME
time=dt.datetime.strptime(olamout[-20:-3],'%Y-%m-%d-%H%M%S')


#============================= CALCULATING THE MASK FOR OCEAN AND LAND ===========================================
glon=gridfile['GLONM']
glat=gridfile['GLATM']
topo=gridfile['TOPM']
topo_data=np.squeeze(griddata(np.stack([glon, glat]).T, topo, outgrids, method='cubic'))
land_mask=np.ma.masked_where(topo_data>=1e-2, topo_data)
ocean_mask=np.ma.masked_where(topo_data<1e-2, topo_data)

#====================================GROUP 1=====================================================================
var_list1, var_list2, var_list3, var_list4 = variable_list()

#input_variables=['SH_C','THETA','ACCPA','dummy1','LAND%VEG_NDVIC','SEA%ALBEDO_DIFFUSE','dummy2','dummy3']

input_variables = [var.strip() for var in  pd.read_csv('input_vars.txt').columns]




id=np.array([[np.where(np.array(lists)==vars)[0] for vars in input_variables]   for lists in np.array([var_list1, var_list2,var_list3, var_list4])])

indices=[]
for var_list_id in range(len(input_variables)):
	indices.append([np.where(np.array([len(var) for var in id[:,var_list_id]])==True)][0][0])

remove_id=np.where(np.array([var.shape[0] for var in np.array(indices)])==0)[0]
save_id=np.setdiff1d(np.arange(len(input_variables)), remove_id)
print(f'The following variable(s) are not available in the input data: {np.array(input_variables)[remove_id]}')

updated_ids=np.delete(id, remove_id,1)
updated_vars=list(np.array(input_variables)[save_id])

group1=list(np.array(input_variables)[np.array([var.shape[0] for var in id[0,:]])==1])
group2=list(np.array(input_variables)[np.array([var.shape[0] for var in id[1,:]])==1])
group3=list(np.array(input_variables)[np.array([var.shape[0] for var in id[2,:]])==1])
group4=list(np.array(input_variables)[np.array([var.shape[0] for var in id[3,:]])==1])

#====================================GROUP 1=====================================================================

variables = {}

if bool(group1):
	latlons =np.vstack((gridfile['GLONW'], gridfile['GLATW'])).T 
	latlons = latlons[1:,:]
	variables = interpolate_grids(olam_out, latlons, outgrids, group1, variables, mask=None)

if bool(group2):
	latlons =np.vstack((landfile['glonwl'], landfile['glatwl'])).T 
	latlons = latlons[1:,:]
	variables = interpolate_grids(olam_out, latlons, outgrids, group2, variables, mask=land_mask.mask)

if bool(group3):
	latlons =np.vstack((seafile['glonws'], seafile['glatws'])).T 
	latlons = latlons[1:,:]
	variables = interpolate_grids(olam_out, latlons, outgrids, group3, variables, mask=ocean_mask.mask)

if bool(group4):
	latlons =np.vstack((gridfile['GLONW'], gridfile['GLATW'])).T 
	latlons = latlons[1:,:]
	variables = interpolate_3grids(olam_out, latlons, outgrids, group4, variables, mask=None)

#====================================METADATA=====================================================================
meta_data=["DZIM","DZIT","DZM","DZT","ZFACIM","ZFACIT","ZFACM","ZFACT","ZM","ZT"]
coordinates_metadata={}
for metadata_id in meta_data:
	coordinates_metadata[metadata_id]=(['level'],np.squeeze(gridfile[metadata_id].values))


#====================================GROUP 2=====================================================================
# LATITUDE, LONGITUDE AND TIME
coordinates={ "lon": (["lon"], lon), "lat": (["lat"], lat),
                "time": (["time"], np.array([time])),
		"ocean_mask": (["lat","lon"], ocean_mask.mask),
		"land_mask": (["lat","lon"], land_mask.mask )}


coordinates.update(coordinates_metadata)

netcdf_file = X.Dataset( variables,
	coords=coordinates)


#===================================================================================================================

var_meta=pd.read_csv('variable_details.txt')

var_meta['var_name']=[vars.strip() for vars in var_meta['var_name'].values]
var_meta['full_name']=[vars.strip() for vars in var_meta['full_name'].values]
var_meta['units']=[vars.strip() for vars in var_meta['units'].values]

for index in list(netcdf_file.keys()):
	#netcdf_file[index].attrs = {'min':np.round(np.min(netcdf_file[index].values),3), 'max':np.round(np.max(netcdf_file[index].values),3) } 
	try:
		id=np.where(var_meta['var_name'].values==index)[0][0]
		netcdf_file[index].attrs = {'full_name':var_meta['full_name'].values[id], 'units':var_meta['units'].values[id], 
						'min':np.round(np.nanmin(netcdf_file[index].values),3), 'max':np.round(np.nanmax(netcdf_file[index].values),3) }	
	except:
		netcdf_file[index].attrs = {'min':np.round(np.nanmin(netcdf_file[index].values),3), 'max':np.round(np.nanmax(netcdf_file[index].values),3) } 

netcdf_file.to_netcdf('test.nc')

















