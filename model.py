import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
# import modelym
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import numpy as np

# torch.manual_seed(701)
# torch.cuda.manual_seed(701)

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(  p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ClassBlock_feature(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock_feature, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        add_block2 = []
        if relu:
            add_block2 += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block2 += [nn.Dropout(p=0.5)]
        add_block2 = nn.Sequential(*add_block2)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.add_block2 = add_block2
        self.classifier = classifier

    def forward(self, x):
        f = self.add_block(x)
        x = self.add_block2(f)
        x = self.classifier(x)
        return f, x

class ClassBlock_BNLinear(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock_BNLinear, self).__init__()
        add_block = []
        add_block += [nn.BatchNorm1d(input_dim)]
        add_block += [nn.Linear(input_dim, class_num)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_classifier)
        self.add_block = add_block

    def forward(self, x):
        x = self.add_block(x)
        return x

class ClassBlock_BNLinear_noModule(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock_BNLinear_noModule, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, class_num)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        return x

class ClassBlock_2dim_BNLinear_noModule(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock_2dim_BNLinear_noModule, self).__init__()

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 2)

        self.bn2 = nn.BatchNorm1d(2)
        self.fc2 = nn.Linear(2, class_num)
        self.fc1.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_classifier)

    def forward(self, x):
        x = self.bn1(x)
        f = self.fc1(x)

        x = self.bn2(f)
        x = self.fc2(x)
        return f, x


class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.

    Args:

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.

        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.

        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.

        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, cuda=True,
                 rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = nn.Parameter(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim), requires_grad=False)

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = nn.Parameter(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim), requires_grad=False)

        if cuda:
            self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.cuda()
            self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.cuda()

    def forward(self, bottom1, bottom2):
        assert bottom1.size(1) == self.input_dim1 and \
            bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        fft1 = torch.fft(torch.cat((sketch_1.unsqueeze(-1), torch.zeros(sketch_1.size()).unsqueeze(-1).cuda()), -1), 1)
        fft2 = torch.fft(torch.cat((sketch_2.unsqueeze(-1), torch.zeros(sketch_2.size()).unsqueeze(-1).cuda()), -1), 1)

        fft1_real = fft1[..., 0]
        fft1_imag = fft1[..., 1]
        fft2_real = fft2[..., 0]
        fft2_imag = fft2[..., 1]

        temp_rr, temp_ii = fft1_real.mul(fft2_real), fft1_imag.mul(fft2_imag)
        temp_ri, temp_ir = fft1_real.mul(fft2_imag), fft1_imag.mul(fft2_real)
        fft_product_real = temp_rr - temp_ii
        fft_product_imag = temp_ri + temp_ir

        cbp_flat = torch.ifft(torch.cat((fft_product_real.unsqueeze(-1), fft_product_imag.unsqueeze(-1)), -1), 1)
        cbp_flat = cbp_flat[..., 0]

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)*self.output_dim

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)
        else:
            cbp = cbp.permute(0,3,1,2)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim ==
               1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense().cuda()


# # |--Linear--|--bn--|--relu--|--Linear--|
# class Class_directly(nn.Module):
#     def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
#         super(ClassBlock, self).__init__()
#         add_block = []
#         add_block += [nn.Linear(input_dim, num_bottleneck)]
#         add_block += [nn.BatchNorm1d(num_bottleneck)]
#         if relu:
#             add_block += [nn.LeakyReLU(0.1)]
#         if dropout:
#             add_block += [nn.Dropout(p=0.5)]
#         add_block = nn.Sequential(*add_block)
#         add_block.apply(weights_init_kaiming)
#
#         classifier = []
#         classifier += [nn.Linear(num_bottleneck, class_num)]
#         classifier = nn.Sequential(*classifier)
#         classifier.apply(weights_init_classifier)
#
#         self.add_block = add_block
#         self.classifier = classifier
#
#     def forward(self, x):
#         x = self.add_block(x)
#         x = self.classifier(x)
#         return x


# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock_direct(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock_direct, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x




class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, conv1, bn1, conv2, bn2, downsample, stride):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
        self.bn2 = bn2
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    # def forward(self, x):
    #     # residual = x
    #
    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu(out)
    #
    #     out = self.conv2(out)
    #     out = self.bn2(out)
    #     out = self.relu(out)
    #
    #     # out = self.conv3(out)
    #     # out = self.bn3(out)
    #
    #     if self.downsample is not None:
    #         residual = self.downsample(x)
    #
    #     # out += residual
    #     out = self.relu(out)
    #
    #     return out

    def forward(self, x):
        # residual = x
        residual_exist = False
        out = self.conv1(x)
        if self.downsample is not None:
            residual = self.bn1(out)
            out = self.relu(residual)
            residual_exist = True

        else:
            out = self.bn1(out)
            out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            residual_exist = True

        if residual_exist:
            out += residual

        out = self.relu(out)

        return out



class ft_net_option_feature(nn.Module):

    def __init__(self, class_num, f_size=512, stride=2, L4id=2):
        super(ft_net_option_feature, self).__init__()
        self.L4id = L4id
        model_ft = models.resnet50(pretrained=True)
        feature_size = f_size

        # avg pooling to global+ pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # model_ft.layer4[2] = Bottleneck(model_ft.layer4[2].conv1, model_ft.layer4[2].bn1, model_ft.layer4[2].conv2,
        #                              model_ft.layer4[2].bn2, model_ft.layer4[2].downsample, model_ft.layer4[2].stride)

        if L4id == 0:
            model_ft.layer4[L4id] = Bottleneck(model_ft.layer4[L4id].conv1, model_ft.layer4[L4id].bn1,
                                               model_ft.layer4[L4id].conv2,
                                               model_ft.layer4[L4id].bn2, None,
                                               model_ft.layer4[L4id].stride)
            if stride == 1:
                model_ft.layer4[0].conv2.stride = (1,1)
        else:
            model_ft.layer4[L4id] = Bottleneck(model_ft.layer4[L4id].conv1, model_ft.layer4[L4id].bn1,
                                               model_ft.layer4[L4id].conv2,
                                               model_ft.layer4[L4id].bn2, model_ft.layer4[L4id].downsample,
                                               model_ft.layer4[L4id].stride)
            if stride == 1:
                model_ft.layer4[0].downsample[0].stride = (1,)
                model_ft.layer4[0].conv2.stride = (1,1)

        # self.pooling = CompactBilinearPooling(feature_size*4, feature_size*4, feature_size)
        self.model = model_ft
        self.classifier = ClassBlock(feature_size, class_num)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        if self.L4id == 0:
            x = self.model.layer4[0](x)
        elif self.L4id == 1:
            x = self.model.layer4[0](x)
            x = self.model.layer4[1](x)
        else:
            x = self.model.layer4(x)
        x = self.model.avgpool(x)
        f0 = torch.squeeze(x)
        x = self.classifier(f0)

        # x = torch.squeeze(x)
        # f0, x = self.classifier(x)
        return f0, x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ft_net_direct(nn.Module):

    def __init__(self, class_num):
        super(ft_net_direct, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        # self.classifier = ClassBlock_2dim_BNLinear_noModule(512, class_num)
        self.classifier = ClassBlock_feature(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        f = torch.squeeze(x)
        f, x = self.classifier(f)
        return f, x



class ft_net_scratch(nn.Module):

    def __init__(self, class_num):
        super(ft_net_scratch, self).__init__()
        model_ft = models.resnet50(pretrained=False)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


class ft_inception_v3_net(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.aux_logits = False
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #
        # x = F.avg_pool2d(x, kernel_size=8)
        # # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # # 1 x 1 x 2048
        # x = x.view(x.size(0), -1)
        # # 2048
        # x = self.fc(x)
        # self.model.fc =nn.Linear(2048, class_num)
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.model.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.model.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.model.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.model.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.model.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.model.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.model.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.model.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.model.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.model.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.model.Mixed_7c(x)
        # 8 x 8 x 2048
        # x = F.avg_pool2d(x, kernel_size=8)
        x = self.model.avgpool(x)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        # x = x.view(x.size(0), -1)
        # 2048
        x = torch.squeeze(x)
        x = self.classifier(x)
        # 1000 (num_classes)

        # x = self.model(x)

        return x
