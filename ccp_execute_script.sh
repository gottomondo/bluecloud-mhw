python ./method-mhw.py --data_path '{{data_path}}' --outputs_path '{{outputs_path}}' --data_source '{{data_source}}' --id_output_type '{{id_output_type}}'  --working_domain '{{working_domain}}' --start_time '{{start_time}}'  --end_time '{{end_time}}' --climatology '{{climatology}}'
if [ $? -ne 0 ]; then
    echo "Error on execution phase" >&2
    exit 1
fi

cp {{ outputs_path }}/* /ccp_data

## Example of execution script after template rendering:
# python ./method-mhw.py --data_path '/inputs' --outputs_path '/outputs' --data_source '["cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021"]' --id_output_type 'mhw_timeseries'  --working_domain '{ "box": [[-4.99, 34, 1, 42]] , "depth_layers": [[0.5,3000] ] }' --start_time '2025-08-24'  --end_time '2025-08-24' --climatology '1987-2021'
# if [ $? -ne 0 ]; then
#     echo "Error on execution phase" >&2
#     exit 1
# fi

# cp /outputs/* /ccp_data