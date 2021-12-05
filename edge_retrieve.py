import cv2
import numpy as np

from PIL import Image

import paddle
import paddle.nn as nn

class HEDBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, paddings, num_convs, with_pool=True):
        super().__init__()
        # VGG Block
        if with_pool:
            pool = nn.MaxPool2D(kernel_size=2, stride=2)
            self.add_sublayer('pool', pool)

        conv1 = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=paddings[0])
        relu = nn.ReLU()

        self.add_sublayer('conv1', conv1)
        self.add_sublayer('relu1', relu)

        for _ in range(num_convs-1):
            conv = nn.Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=paddings[_+1])
            self.add_sublayer(f'conv{_+2}', conv)
            self.add_sublayer(f'relu{_+2}', relu)

        self.layer_names = [name for name in self._sub_layers.keys()]

        # Socre Layer
        self.score = nn.Conv2D(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        for name in self.layer_names:
            input = self._sub_layers[name](input)
        return input, self.score(input)

class HED_Caffe(nn.Layer):
    def __init__(self,
                 channels=[3, 64, 128, 256, 512, 512],
                 nums_convs=[2, 2, 3, 3, 3],
                 paddings=[[35, 1], [1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                 crops=[34, 35, 36, 38, 42],
                 with_pools=[False, True, True, True, True]):
        super().__init__()
        '''
        Caffe HED model re-implementation in Paddle.

        This model is based on the official Caffe pre-training model. 
        The inference results of this model are very close to the official implementation in Caffe.
        Pytorch and Paddle's Bilinear Upsampling are not completely equivalent to Caffe's DeConvolution with Bilinear, so Transpose Convolution with Bilinear is used instead.
        In the official Caffe pre-training model, the padding parameter value of the first convolution layer is equal to 35, so the feature map needs to be cropped. 
        The crop parameters refer to the code implementation by XWJABC. The code link: https://github.com/xwjabc/hed/blob/master/networks.py#L55.
        '''
        assert (len(channels) - 1) == len(nums_convs), '(len(channels) -1) != len(nums_convs).'

        self.crops = crops

        # HED Blocks
        for index, num_convs in enumerate(nums_convs):
            block = HEDBlock(in_channels=channels[index], out_channels=channels[index+1], paddings=paddings[index], num_convs=num_convs, with_pool=with_pools[index])
            self.add_sublayer(f'block{index+1}', block)

        self.layer_names = [name for name in self._sub_layers.keys()]

        # Upsamples
        for index in range(2, len(nums_convs)+1):
            upsample = nn.Conv2DTranspose(in_channels=1, out_channels=1, kernel_size=2**index, stride=2**(index-1), bias_attr=False)
            upsample.weight.set_value(self.bilinear_kernel(1, 1, 2**index))
            upsample.weight.stop_gradient = True
            self.add_sublayer(f'upsample{index}', upsample)

        # Output Layers
        self.out = nn.Conv2D(in_channels=len(nums_convs), out_channels=1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        h, w = input.shape[2:]
        scores = []
        for index, name in enumerate(self.layer_names):
            input, score = self._sub_layers[name](input)
            if index > 0:
                score = self._sub_layers[f'upsample{index+1}'](score)

            score = score[:, :, self.crops[index]: self.crops[index] + h, self.crops[index]: self.crops[index] + w]
            scores.append(score)

        output = self.out(paddle.concat(scores, 1))
        return self.sigmoid(output)

    @staticmethod
    def bilinear_kernel(in_channels, out_channels, kernel_size):
        '''
        return a bilinear filter tensor
        '''
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return paddle.to_tensor(weight, dtype='float32')


def hed_caffe(pretrained=True, **kwargs):
    model = HED_Caffe(**kwargs)
    if pretrained:
        pdparams = paddle.load('hed_pretrained_bsds.pdparams')
        model.set_dict(pdparams)
    return model

def preprocess(img):
    img = img.astype('float32')
    # img -= np.asarray([104.00698793, 116.66876762, 122.67891434], dtype='float32')
    img = img.transpose(2, 0, 1)
    img = img[None, ...]
    return paddle.to_tensor(img, dtype='float32')

def postprocess(outputs):
    results = paddle.clip(outputs, 0, 1)
    results = paddle.squeeze(results, 1)
    results *= 255.0
    results = results.cast('uint8')
    return results.numpy()


if __name__ == '__main__':
    model = hed_caffe(pretrained=True)
    img = cv2.imread('./save/munster_000025_000019_leftImg8bit_pre_mask_1024x2048.png')
    img_tensor = preprocess(img)
    outputs = model(img_tensor)
    results = postprocess(outputs)

    show_img = np.concatenate([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(results[0], cv2.COLOR_GRAY2RGB)], 1)
    Image.fromarray(show_img)