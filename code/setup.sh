#!/usr/bin/env bash

set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python -c 'import torch; print("Keeping existing PyTorch:", torch.__version__); assert torch.cuda.is_available(), "Activate the working CUDA environment first"; print("GPU:", torch.cuda.get_device_name(0))'
python -m pip install -r requirements.txt
python check_cn6_nofusion.py
mkdir -p logs
python -m pip freeze > logs/environment.txt


