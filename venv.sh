#/bin/bash

virtualenv --no-site-packages -p python3 pyenv
source pyenv/bin/activate
pip install -r requirements.txt
