# method mhw
#

import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="MHW detection and mapping")
# parser.add_argument("--ccpimage", type=str, required=True, help="Docker image")
parser.add_argument("--repository", type=str, required=False, default="", help="Repository path")
parser.add_argument("--data_path", type=str, required=False, help="Path to input data")
parser.add_argument("--outputs_path", type=str, required=False, help="Path to output data")
parser.add_argument("--data_source", type=str, required=True, help="Dataset ID")
parser.add_argument("--id_output_type", type=str, required=True, help="Output type (e.g., mhw_timeseries)")
parser.add_argument("--working_domain", type=str, required=True, help="Working domain")
parser.add_argument("--start_time", type=str, required=True, help="Start date (YYYY-MM-DD)")
parser.add_argument("--end_time", type=str, required=False, help="End date (YYYY-MM-DD)")

args = parser.parse_args()

# Assign variables
# ccpimage = args.ccpimage
repository = args.repository
data_path = args.data_path
outputs_path = args.outputs_path
data_source = args.data_source
id_output_type = args.id_output_type
working_domain = args.working_domain
start_time = datetime.strptime(args.start_time, "%Y-%m-%d")
if args.end_time is not None:
    end_time = datetime.strptime(args.end_time, "%Y-%m-%d")
else:
    end_time = None  # or set a default value if you want

# Save all arguments to outputs.txt in the outputs_path directory
with open(f"{outputs_path}/inputs.txt", "w") as f:
    # f.write(f"ccpimage: {ccpimage}\n")
    f.write(f"repository: {repository}\n")
    f.write(f"data_path: {data_path}\n")
    f.write(f"outputs_path: {outputs_path}\n")
    f.write(f"data_source: {data_source}\n")
    f.write(f"id_output_type: {id_output_type}\n")
    f.write(f"working_domain: {working_domain}\n")
    f.write(f"start_time: {start_time.strftime('%Y-%m-%d')}\n")
    if args.end_time is not None:
        f.write(f"end_time: {end_time.strftime('%Y-%m-%d')}\n")