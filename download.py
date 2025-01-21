import os, sys
import requests

args = sys.argv

data_source = args[1]
data_path = args[2]


CMEMS_datasets_options = {'cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021':{
	'varname':  "analysed_sst", "prod_type": 'REP',
	'grid_file':"...",
	'clim_file':"...",
	'grid_url':"",
	'clim_url':""},
         
'SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2': {'varname':  "analysed_sst", "prod_type": 'NRT',
	'grid_file':"...",
	'clim_file':"...",
	'grid_url':"",
	'clim_url':""}
}

print("Selected Copernicus Marine Service (CMEMS) dataset: " + data_source)

grid_file = CMEMS_datasets_options[data_source]['grid_file']
clim_file = CMEMS_datasets_options[data_source]['clim_file']

grid_url = CMEMS_datasets_options[data_source]['grid_url']
clim_url = CMEMS_datasets_options[data_source]['clim_url']

response = requests.get(grid_url, headers=headers)
with open(grid_file, 'wb') as f:
    f.write(response.content)

response = requests.get(clim_url, headers=headers)
with open(clim_file, 'wb') as f:
    f.write(response.content)

