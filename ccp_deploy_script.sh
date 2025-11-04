mkdir -p {{ data_path }} {{outputs_path}}
git clone {{ repository }} mhw
cd mhw/app
python ./download.py '{{data_source}}' '{{data_path}}' '{{climatology}}'
if [ $? -ne 0 ]; then
    echo "Error on download phase" >&2
    exit 0
fi

## Example of execution script after template rendering:
# mkdir -p /inputs /outputs
# git clone https://github.com/gottomondo/bluecloud-mhw.git mhw
# cd mhw/app
# python ./download.py '["cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021"]' '/inputs' '1987-2021'
# if [ $? -ne 0 ]; then
#     echo "Error on download phase" >&2
#     exit 0
# fi