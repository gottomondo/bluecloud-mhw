# CMEMS datasets metadata
# take care: climatology and MFS on different grids
CMEMS_datasets_options = {
    'cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021': {
        'label': 'Mediterranean SAT REP',
        'varname': "analysed_sst",
        'prod_type': 'L4-REP',
        'grid_file': "cell_areas_CMS.nc",
        'area_file': "cell_areas_CMS.nc",
        'grid_url': "https://data.d4science.net/tgUFc",
        'area_url': "https://data.d4science.net/tgUFc",
        'clim_file': {
            "1987-2021": {
                "file": "CMS_SST_Climatology_19872021.nc",
                "url": "https://data.d4science.net/YB7c"
            },
            "1993-2022": {
                "file": "MED-L4-REP_cmems_Climatology_19932022.nc",
                "url": "https://data.d4science.net/LjU98"
            } 
        },
        'region_folder': 'prev'
    },
    'SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2': {
        'label': 'Mediterranean SAT NRT',
        'varname': "analysed_sst",
        'prod_type': 'L4-NRT',
        'grid_file': "cell_areas_CMS_NRT.nc",
        'area_file': "cell_areas_CMS_NRT.nc",
        'grid_url': "https://data.d4science.net/kyvw",
        'area_url': "https://data.d4science.net/kyvw",
        'clim_file': {
            "1987-2021": {
                "file": "CMS_SST_Climatology_19872021_rect00625.nc",
                "url": "https://data.d4science.net/Sx86"
            },
            "1993-2022": {
                "file": "MED-L4-REP_cmems_Climatology_19932022_interp-NRT.nc",
                "url": "https://data.d4science.net/oxfGr"
            }
        },
        'region_folder': '2024'
    },
    'cmems_mod_med_phy-tem_anfc_4.2km_P1D-m': {
        'label': 'Mediterranean Forecast Model NRT',
        'varname': "thetao",
        'prod_type': 'MFS',
        'grid_file': "cell_areas_MFS_CMS.nc",
        'area_file': "cell_areas_MEDREA.nc",
        'grid_url': "https://data.d4science.net/rEUwc",
        'area_url': "https://data.d4science.net/JD3Io",
        'clim_file': {
            "1987-2021": {
                "file": "MEDREA_native_Climatology_depth0_19872021.nc",
                "ur": "https://data.d4science.net/KcBLU"
            },
            "1993-2022": {
                "file": "MEDREA_native_Climatology_depth0_19932022.nc",
                "url": ""
            },
        },
        'region_folder': 'MFS_CMS'
    }
}

# MHW categories info for plotting
category_info = {1: {'name': 'Moderate', 'color': '#FFD700'},  # Yellow
                 2: {'name': 'Strong',   'color': '#FF6F00'},  # Brighter orange
                 3: {'name': 'Severe',   'color': '#D32F2F'},  # Strong Red
                 4: {'name': 'Extreme',  'color': '#5D2611'},  # Dark Brown}
                }

# Regions strings for titles and labels
regions_strings = {'MedWhole':'Mediterranean Sea',
                   'AdriaticSouth':'South Adriatic Sea',
                   'Adriatic':'Adriatic Sea',
                   'LigurianSea':'Ligurian Sea',
                    }