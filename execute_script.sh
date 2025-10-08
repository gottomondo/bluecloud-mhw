# Execute script: runs your algorithm with input parameters
# Example:
# python ./method-mhw.py  --outputs_path "." --data_source '[ "DS" ]' --id_output_type "id_output_type" --working_domain '{ "box": [[-4.99, 34, 1, 42]] , "depth_layers": [[0.5,3000] ] }' --start_time "2025-10-08"  
python ./method-mhw.py --repository {{repository}} --data_path {{data_path}} --outputs_path {{outputs_path}} --data_source {{data_source}} --id_output_type {{id_output_type}}  --working_domain {{working_domain}} --start_time {{start_time}}  --end_time {{end_time}}
cp /outputs/* /ccp_data