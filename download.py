import os, sys
import requests
import time
from config import CMEMS_datasets_options

args = sys.argv

script_name = args[0]

if len(sys.argv) != 3:
    print("Usage: python {} {} {}".format(script_name, "<data_source>", "<data_path>"))
    print("Example: python {} SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2 tmp".format(script_name))
    sys.exit(1)
    
# inputs
data_source = args[1]
data_path = args[2]

def download_file(url, output_file, headers=None, max_retries=5, delay=3):
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries} - Downloading: {output_file}")
            t0 = time.time()
            response = requests.get(url, headers=headers, timeout=10) 
            # Check that the request was successful (status code 200)
            response.raise_for_status()
            # Save the contents to a binary file
            with open(output_file, 'wb') as f:
                f.write(response.content)  
            print(f"\tDone ({time.time() - t0:.1f}s).")
            print(f"File downloaded successfully: {output_file}") 
            return True
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error during download: {http_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error: {req_err}")
        except Exception as e:
            print(f"General error: {e}")

        attempt += 1
        if attempt < max_retries:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print(f"Max retries reached. Failed to download: {output_file}")
            return False

grid_file = CMEMS_datasets_options[data_source]['grid_file']
clim_file = CMEMS_datasets_options[data_source]['clim_file']

grid_url = CMEMS_datasets_options[data_source]['grid_url']
clim_url = CMEMS_datasets_options[data_source]['clim_url']

print("Selected dataset: " + data_source)
print("Selected path: " + data_path)

headers = {"User-Agent": "bc2026-vlab4-mei/1.0"}

if not download_file(grid_url, os.path.join(data_path, grid_file), headers=headers, max_retries=3, delay=5):
    print("Aborting script due to failed grid file download.")
    sys.exit(1)

if not download_file(clim_url, os.path.join(data_path, clim_file), headers=headers, max_retries=3, delay=5):
    print("Aborting script due to failed clim file download.")
    sys.exit(1)
