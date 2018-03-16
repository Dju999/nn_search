#/bin/bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
virtualenv --no-site-packages -p python3 $DIR'/pyenv'
pip install -r $DIR'/requirements.txt'
