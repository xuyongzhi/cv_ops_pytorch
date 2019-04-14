**xyz April 2019**

# References
- ROI, NMS:  https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark/csrc
- RoI align rotate:  https://github.com/pytorch/pytorch/tree/3796cbaf7ab4c4c30cb99191d55a8b9c50b398dc/caffe2/operators

# Environment
- pytorch 1.0
- cuda 9.0
- python 3.7
- gcc 5.5

# Building
./build.sh (build so only, not installed)
./cp_so.sh

just ignore errors for rm non exist directories


# Test
## test build so
``` bash
cd layers
python roi_align.py
python roi_align_rotated.py
```
## test cpp directly
```
cd test
not implemented yet
```


