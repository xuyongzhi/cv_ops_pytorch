# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import cv_ops_lib
print(cv_ops_lib.__file__)

class _ROIAlignRotated(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        # input: [4, 256, 304, 200]
        # roi: [171, 5]
        # spatial_scale: 0.25
        # output_size: [7,7]
        # sampling_ratio: 2
        #output = cv_ops_lib.roi_align_forward(
        output = cv_ops_lib.roi_align_rotated_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        ) # [171, 256, 7, 7]
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        #grad_input = cv_ops_lib.roi_align_backward(
        grad_input = cv_ops_lib.roi_align_rotated_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None


roi_align = _ROIAlignRotated.apply


class ROIAlignRotated(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        '''
        output_size:[pooled_height, pooled_width]
        spatial_scale: size_of_map/size_of_original_image
        sampling_ratio: how many points to use for bilinear_interpolate
        '''
        super(ROIAlignRotated, self).__init__()
        self.output_size = output_size # (7,7)
        self.spatial_scale = spatial_scale # 0.25
        self.sampling_ratio = sampling_ratio # 2

    def forward(self, input, rois):
        '''
        input: [batch_size, feature, w, h]
        rois: [n,5] [batch_ind, center_w, center_h, roi_width, roi_height, theta]
            theta unit: degree, anti-clock wise is positive
        '''
        assert rois.shape[1] == 6
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr



if __name__ == '__main__':
    # note: output_size: [h,w],  rois: [w,h]
    # order is different

    align_roi = ROIAlignRotated((1, 3), 1, 2)
    feat = torch.arange(64).view(1, 1, 8, 8).float()
    # Note: first element is batch_idx
    rois = torch.tensor([
          [0, 3,3, 3,1, 0],
          [0, 3,3, 3,1, 90],
          [0, 3,3, 3,1, -90],
          [0, 3,3, 3,1, 30],
          [0, 3,3, 3,1, 60],
          ], dtype=torch.float32).view(-1, 6)

    print(f'feat:\n{feat}\nrois:\n{rois}')

    print('------------test on cpu------------')
    feat.requires_grad = False
    if False:
      out = align_roi(feat, rois)
      print(out)
      print('cpu version do not support backward')
    #out.sum().backward()
    #print(feat.grad)

    if torch.cuda.is_available():
        print('------------test on gpu------------')
        feat = feat.detach().cuda()
        rois = rois.cuda()
        feat.requires_grad = True
        out = align_roi(feat, rois)
        print(out)
        temp = out.sum()
        temp.backward()
        print(feat.grad)
    else:
        print('You device have not a GPU')

