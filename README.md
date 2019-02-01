
name: "VGG_VOC0712_SSD_256x128_deploy"
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 192
  dim: 192
}
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv1_1_bn"
  type: "BatchNorm"
  bottom: "conv1_1"
  top: "conv1_1"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
}
layer {
  name: "conv1_1_scale"
  type: "Scale"
  bottom: "conv1_1"
  top: "conv1_1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu1_1"
  type: "ReLU"
  bottom: "conv1_1"
  top: "conv1_1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1_1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2_1"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv2_1_bn"
  type: "BatchNorm"
  bottom: "conv2_1"
  top: "conv2_1"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
}
layer {
  name: "conv2_1_scale"
  type: "Scale"
  bottom: "conv2_1"
  top: "conv2_1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu2_1"
  type: "ReLU"
  bottom: "conv2_1"
  top: "conv2_1"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2_1"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv3_1"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv3_1_bn"
  type: "BatchNorm"
  bottom: "conv3_1"
  top: "conv3_1"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
}
layer {
  name: "conv3_1_scale"
  type: "Scale"
  bottom: "conv3_1"
  top: "conv3_1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu3_1"
  type: "ReLU"
  bottom: "conv3_1"
  top: "conv3_1"
}

layer {
  name: "pool3"
  type: "Pooling"
  bottom: "conv3_1"
  top: "pool3"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv4_1"
  type: "Convolution"
  bottom: "pool3"
  top: "conv4_1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_1_bn"
  type: "BatchNorm"
  bottom: "conv4_1"
  top: "conv4_1"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
}
layer {
  name: "conv4_1_scale"
  type: "Scale"
  bottom: "conv4_1"
  top: "conv4_1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu4_1"
  type: "ReLU"
  bottom: "conv4_1"
  top: "conv4_1"
}
layer {
  name: "conv4_2"
  type: "Convolution"
  bottom: "conv4_1"
  top: "conv4_2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_2_bn"
  type: "BatchNorm"
  bottom: "conv4_2"
  top: "conv4_2"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
}
layer {
  name: "conv4_2_scale"
  type: "Scale"
  bottom: "conv4_2"
  top: "conv4_2"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "relu4_2"
  type: "ReLU"
  bottom: "conv4_2"
  top: "conv4_2"
}

layer {
  name: "pool4"
  type: "Pooling"
  bottom: "conv4_2"
  top: "pool4"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}

layer {
  name: "conv5_512_up"
  type: "Deconvolution"
  bottom: "pool4"
  top: "conv5_512_up"
  convolution_param {
    kernel_size: 4 
    stride: 2
    num_output: 512
    group: 512
    pad: 1
    weight_filler: { type: "bilinear" } 
    bias_term: false
  }
  param { lr_mult: 0 decay_mult: 0 }
}
# Crop conv5_3
layer {
  name: "conv5_512_crop"
  type: "Crop"
  bottom: "conv5_512_up"
  bottom: "conv4_1"
  top: "conv5_512_crop"
  crop_param {
    axis: 2
    offset: 0
  }
}
layer {
  name: "conv4_fuse"
  type: "Eltwise"
  bottom: "conv5_512_crop"
  bottom: "conv4_1"
  top: "conv4_1_SUM"
  eltwise_param {
    operation: SUM
  }
}

layer {
  name: "inception_3a/1x1"
  type: "Convolution"
  bottom: "conv4_1_SUM"
  top: "inception_3a/1x1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "inception_3a/1x1_bn"
  type: "BatchNorm"
  bottom: "inception_3a/1x1"
  top: "inception_3a/1x1"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
}
layer {
  name: "inception_3a/1x1_scale"
  type: "Scale"
  bottom: "inception_3a/1x1"
  top: "inception_3a/1x1"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "inception_3a/relu_1x1"
  type: "ReLU"
  bottom: "inception_3a/1x1"
  top: "inception_3a/1x1"
}
layer {
  name: "inception_3a/3x3_reduce"
  type: "Convolution"
  bottom: "conv4_1_SUM"
  top: "inception_3a/3x3_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "inception_3a/3x3_reduce_bn"
  type: "BatchNorm"
  bottom: "inception_3a/3x3_reduce"
  top: "inception_3a/3x3_reduce"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
}
layer {
  name: "inception_3a/3x3_reduce_scale"
  type: "Scale"
  bottom: "inception_3a/3x3_reduce"
  top: "inception_3a/3x3_reduce"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "inception_3a/relu_3x3_reduce"
  type: "ReLU"
  bottom: "inception_3a/3x3_reduce"
  top: "inception_3a/3x3_reduce"
}
layer {
  name: "inception_3a/3x3"
  type: "Convolution"
  bottom: "inception_3a/3x3_reduce"
  top: "inception_3a/3x3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 4
    kernel_size: 3
    dilation: 4
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "inception_3a/3x3_bn"
  type: "BatchNorm"
  bottom: "inception_3a/3x3"
  top: "inception_3a/3x3"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
}
layer {
  name: "inception_3a/3x3_scale"
  type: "Scale"
  bottom: "inception_3a/3x3"
  top: "inception_3a/3x3"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "inception_3a/relu_3x3"
  type: "ReLU"
  bottom: "inception_3a/3x3"
  top: "inception_3a/3x3"
}
layer {
  name: "inception_3a/5x5_reduce"
  type: "Convolution"
  bottom: "conv4_1_SUM"
  top: "inception_3a/5x5_reduce"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "inception_3a/5x5_reduce_bn"
  type: "BatchNorm"
  bottom: "inception_3a/5x5_reduce"
  top: "inception_3a/5x5_reduce"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
}
layer {
  name: "inception_3a/5x5_reduce_scale"
  type: "Scale"
  bottom: "inception_3a/5x5_reduce"
  top: "inception_3a/5x5_reduce"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "inception_3a/relu_5x5_reduce"
  type: "ReLU"
  bottom: "inception_3a/5x5_reduce"
  top: "inception_3a/5x5_reduce"
}
layer {
  name: "inception_3a/5x5"
  type: "Convolution"
  bottom: "inception_3a/5x5_reduce"
  top: "inception_3a/5x5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 128
    pad: 8
    kernel_size: 5
    dilation: 4
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "inception_3a/5x5_bn"
  type: "BatchNorm"
  bottom: "inception_3a/5x5"
  top: "inception_3a/5x5"
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
  param {
    lr_mult: 1.0
  }
}
layer {
  name: "inception_3a/5x5_scale"
  type: "Scale"
  bottom: "inception_3a/5x5"
  top: "inception_3a/5x5"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 1.0
  }
  scale_param {
    bias_term: true
  }
}
layer {
  name: "inception_3a/relu_5x5"
  type: "ReLU"
  bottom: "inception_3a/5x5"
  top: "inception_3a/5x5"
}
layer {
  name: "inception_3a/output"
  type: "Concat"
  bottom: "inception_3a/1x1"
  bottom: "inception_3a/3x3"
  bottom: "inception_3a/5x5"
  top: "inception_3a/output"
}

layer {
  name: "conv4_1_norm"
  type: "Normalize"
  bottom: "inception_3a/output"
  top: "conv4_1_norm"
  norm_param {
    across_spatial: false
    scale_filler {
      type: "constant"
      value: 20
    }
    channel_shared: false
  }
}


layer {
  name: "conv4_1_norm_mbox_loc"
  type: "Convolution"
  bottom: "conv4_1_norm"
  top: "conv4_1_norm_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_1_norm_mbox_loc_perm"
  type: "Permute"
  bottom: "conv4_1_norm_mbox_loc"
  top: "conv4_1_norm_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_1_norm_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv4_1_norm_mbox_loc_perm"
  top: "conv4_1_norm_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_1_norm_mbox_conf_r1"
  type: "Convolution"
  bottom: "conv4_1_norm"
  top: "conv4_1_norm_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_1_norm_mbox_conf_perm_r1"
  type: "Permute"
  bottom: "conv4_1_norm_mbox_conf"
  top: "conv4_1_norm_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_1_norm_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv4_1_norm_mbox_conf_perm"
  top: "conv4_1_norm_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_1_norm_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv4_1_norm"
  bottom: "data"
  top: "conv4_1_norm_mbox_priorbox"
  prior_box_param {
    min_size: 30.0
    max_size: 60.0
    aspect_ratio: 2.0
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    offset: 0.5
  }
}


layer {
  name: "mbox_conf_reshape"
  type: "Reshape"
  bottom: "conv4_1_norm_mbox_conf_flat"
  top: "mbox_conf_reshape"
  reshape_param {
    shape {
      dim: 0
      dim: -1
      dim: 3
    }
  }
}
layer {
  name: "mbox_conf_softmax"
  type: "Softmax"
  bottom: "mbox_conf_reshape"
  top: "mbox_conf_softmax"
  softmax_param {
    axis: 2
  }
}
layer {
  name: "mbox_conf_flatten"
  type: "Flatten"
  bottom: "mbox_conf_softmax"
  top: "mbox_conf_flatten"
  flatten_param {
    axis: 1
  }
}

layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "conv4_1_norm_mbox_loc_flat"
  bottom: "mbox_conf_flatten"
  bottom: "conv4_1_norm_mbox_priorbox"
  top: "detection_out"
  include {
    phase: TEST
  }
  detection_output_param {
    num_classes: 3
    share_location: true
    background_label_id: 0
    nms_param {
      nms_threshold: 0.45
      top_k: 400
    }
    code_type: CENTER_SIZE
    keep_top_k: 200
    confidence_threshold: 0.01
  }
}
