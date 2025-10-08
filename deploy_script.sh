# Deploy script: runs before execution. Use it to:
# - Clone your repo
# - Install dependencies
# - Download data

git clone {{ repository }} mhw
cd mhw
# pip install -r requirements.txt
# mkdir {{ data_path }}
# python download.py {{ data_source }} {{ data_path }}
# if [ $? -ne 0 ]; then
#  echo "error: download of starting files failed"
#  exit 1
# fi