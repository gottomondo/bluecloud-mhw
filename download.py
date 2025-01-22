import os, sys
import requests
import time

args = sys.argv

if len(sys.argv) != 3:
	print("Usage: python download.py <data_source> <data_path>")
	sys.exit(1)
    
# inputs
data_source = args[1]
data_path = args[2]

CMEMS_datasets_options = {'cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021':{'varname':  "analysed_sst", "prod_type": 'REP',
                                                                        #  'grid_file':"prev/cell_areas_CMS.nc",
                                                                         'grid_file':"cell_areas_CMS.nc",
                                                                         'grid_url': 'https://data.d4science.net/RNaQ', 
                                                                         'clim_file':"CMS_SST_Climatology_19872021.nc",
                                                                         'clim_url': 'https://data.d4science.net/YB7c'},
                        'SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2': {'varname':  "analysed_sst", "prod_type": 'NRT',
                                                                        #  'grid_file':"2024/cell_areas_CMS_NRT.nc",
                                                                         'grid_file':"cell_areas_CMS_NRT.nc",
                                                                         'grid_url': 'https://data.d4science.net/kyvw',
                                                                         'clim_file':"CMS_SST_Climatology_19872021_rect00625.nc",
                                                                         'clim_url': 'https://data.d4science.net/Sx86'},
}

def download_file(url, output_file, headers=None):
    try:
        print("\nDownloading: " + output_file)
        t0 = time.time()
        response = requests.get(url, headers=headers, timeout=10) 
        # Check that the request was successful (status code 200)
        response.raise_for_status()
        # Save the contents to a binary file
        with open(output_file, 'wb') as f:
            f.write(response.content)  
        print(f"\tDone ({time.time() - t0:.1f}s).")
        print(f"File downloaded successfully: {output_file}") 
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error during download: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error: {req_err}")
    except Exception as e:
        print(f"General error: {e}")


grid_file = CMEMS_datasets_options[data_source]['grid_file']
clim_file = CMEMS_datasets_options[data_source]['clim_file']

grid_url = CMEMS_datasets_options[data_source]['grid_url']
clim_url = CMEMS_datasets_options[data_source]['clim_url']

print("Selected dataset: " + data_source)
print("Selected path: " + data_path)

headers = {"User-Agent": "bc2026-vlab4-mei/1.0"}

download_file(grid_url, os.path.join(data_path, grid_file), headers=headers)
download_file(clim_url, os.path.join(data_path, clim_file), headers=headers)



