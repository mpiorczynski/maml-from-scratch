#!/bin/bash
if [ $# -lt 1 ]; then
  echo "Usage: $0 <script> [arg1] [arg2]..."
  exit 1
fi

script_name=$1
extension="${script_name##*.}"
# shift arguments to leave only the command line arguments for the script
shift 

source .env
eval "$(conda shell.bash hook)"
conda activate maml-from-scratch
# export  PYTHONPATH="${PYTHONPATH}:src"

if [ "$extension" == "py" ]; then
    python $script_name "$@"
elif [ "$extension" == "sh" ]; then
    bash $script_name "$@"
else
    echo "Unsupported file format: .$extension"
    exit 1
fi