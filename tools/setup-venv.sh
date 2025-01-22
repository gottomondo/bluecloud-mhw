#!/bin/bash
# setup/update venv
# you can update req_file for installing new modules
# 

# venv name
venv_name=${1:-venv}

# python version
py_ver=${2:-python3}

# requirements file
req_file=${3:-requirements.txt}

if [ ! -d $venv_name ]; then
    echo "VirtualEnv installation..."
    virtualenv -p $py_ver $venv_name
fi

if [ -f $req_file ]; then
    echo "Requirements installation..."
    source ./$venv_name/bin/activate
    pip install --upgrade pip
    pip install -r $req_file
    deactivate
fi