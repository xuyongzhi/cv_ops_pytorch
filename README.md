**xyz April 2019**

# Building
./build.sh
./cp_so.sh

just ignore errors for rm non exist directories


# Test
``` bash
cd layers
python roi_align.py
```

# References
# # ROI, NMS
from:  https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark/csrc

## RoI align rotate
code from https://github.com/pytorch/pytorch/tree/3796cbaf7ab4c4c30cb99191d55a8b9c50b398dc/caffe2/operators

