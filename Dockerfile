# To build:
# docker build -t bluecloud-method-mhw .
# To start:
# ./docker_run.sh

FROM python:3.10.11

RUN python3 -m pip install xarray \
numpy \
matplotlib \
netcdf4 \
cartopy \
scipy \
wget

COPY Maps.py /data/Maps.py
COPY wavesnspikes.py /data/wavesnspikes.py

# RUN wget https://data.d4science.net/3po4 -O /data/NRT.nc
# RUN wget https://data.d4science.net/6LZi -O /data/CELL.nc
# RUN wget https://data.d4science.net/q3U8 -O /data/CLIM.nc

COPY data/20240731000000-GOS-L4_GHRSST-SSTfnd-OISST_HR_NRT-MED-v02.0-fv03.0.nc /data/NRT.nc
COPY data/cell_areas_CMS_NRT.nc /data/CELL.nc
COPY data/CMS_SST_Climatology_19872021_rect00625.nc /data/CLIM.nc

CMD [ "python" , "/data/Maps.py" ]
