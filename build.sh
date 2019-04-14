#/usr/bin/bash

rm -r build

./clean.sh
python setup.py build

