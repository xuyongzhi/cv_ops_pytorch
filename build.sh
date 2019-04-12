#/usr/bin/bash

rm -r build
#rm -r cv_ops_pytorch.egg-info
#rm -r dist
#rm  r /home/z/miniconda3/envs/m/lib/python3.7/site-packages/cv_ops_pytorch-0.1-py3.7-linux-x86_64.egg


./clean.sh
python setup.py build

#cp ./build/lib.linux-x86_64-3.7/*.so ./layers
