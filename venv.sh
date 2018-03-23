#/bin/bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
virtualenv --no-site-packages -p python3 $DIR'/pyenv'
source $DIR'/pyenv/bin/activate'
$DIR'/pyenv/bin/pip' install -r $DIR'/requirements.txt'
