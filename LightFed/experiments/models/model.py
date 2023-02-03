import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
# from transformers import BertModel


# CONFIGS_ = {
#     # convolution layer (feature extraction) configuration, classifier configuration, input_channel, n_class, hidden_dim, latent_dim
#     'cifar': ([16, 'M', 32, 'M', 'F'], 3, 10, 2048, 64),
#     'cifar100-c25': ([32, 'M', 64, 'M', 128, 'F'], 3, 25, 128, 128),
#     'cifar100-c30': ([32, 'M', 64, 'M', 128, 'F'], 3, 30, 2048, 128),
#     'cifar100-c50': ([32, 'M', 64, 'M', 128, 'F'], 3, 50, 2048, 128),
#
#     'emnist': ([6, 16, 'F'], 1, 26, 784, 32),
#     'MNIST': ([6, 16, 'F'], 1, 10, 784, 32),
#     'mnist_cnn1': ([6, 'M', 16, 'M', 'F'], 1, 10, 64, 32),
#     'mnist_cnn2': ([16, 'M', 32, 'M', 'F'], 1, 10, 128, 32),
#     'celeb': ([16, 'M', 32, 'M', 64, 'M', 'F'], 3, 2, 64, 32)
# }
#
# # temporary roundabout to evaluate sensitivity of the generator
# GENERATORCONFIGS = {
#     # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
#     'cifar': (512, 32, 3, 10, 64),
#     'celeb': (128, 32, 3, 2, 32),
#     'MNIST': (256, 32, 1, 10, 32),
#     'mnist-cnn0': (256, 32, 1, 10, 64),
#     'mnist-cnn1': (128, 32, 1, 10, 32),
#     'mnist-cnn2': (64, 32, 1, 10, 32),
#     'mnist-cnn3': (64, 32, 1, 10, 16),
#     'emnist': (256, 32, 1, 26, 32),
#     'emnist-cnn0': (256, 32, 1, 26, 64),
#     'emnist-cnn1': (128, 32, 1, 26, 32),
#     'emnist-cnn2': (128, 32, 1, 26, 16),
#     'emnist-cnn3': (64, 32, 1, 26, 32),
# }

CONFIGS_ = {
    # convolution layer (feature extraction) configuration, classifier configuration, input_channel, n_class, latent_dim(表示特征提取fllaten后的维度)
    'MNIST': ([6, 'M', 16, 'M', 'F'], [32], 1, 10, 64),
    'FMNIST': ([16, 'M', 32, 'M', 'F'], [64], 1, 10, 128),
    'EMNIST': ([16, 'M', 32, 'M', 'F'], [128], 1, 10, 128),
    # 'EMNIST': ([16, 'M', 32, 'M', 'F'], [128], 1, 26, 128),
    # 'EMNIST': ([6, 16, 'F'], [128, 64], 1, 37, 784),
    'CIFAR-10': ([16, 'M', 32, 'M', 'F'], [128], 3, 10, 128)

}

CLASSIFERCONFIGS = {
    # convolution layer (feature extraction) configuration, classifier configuration, input_channel, n_class, latent_dim(表示特征提取fllaten后的维度)
    'MNIST': ([], [32], 1, 10, 64),
    'FMNIST': ([], [64], 1, 10, 128),
    'EMNIST': ([], [128], 1, 10, 128),
    # 'EMNIST': ([], [128], 1, 26, 128),
    # 'EMNIST': ([], [128, 64], 1, 37, 784),
    'CIFAR-10':([], [128], 3, 10, 128),

    'CIFAR-100':([], [128], 3, 100, 256)
}

DISCRIMINATORCONFIGS = {
    # convolution layer (feature extraction) configuration, classifier configuration, input_channel, n_class, latent_dim(表示特征提取fllaten后的维度)
    'MNIST': ([], [32], 1, 1, 64),
    'FMNIST': ([], [64], 1, 1, 128),
    'EMNIST': ([], [128], 1, 1, 128),
    # 'EMNIST': ([], [128], 1, 1, 128),
    # 'EMNIST': ([], [128], 1, 1, 784),
    'CIFAR-10':([], [128], 3, 1, 128),

    'CIFAR-100':([], [128], 3, 1, 256)
}

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'MNIST': ([512, 128], 64, 1, 10, 64),
    'FMNIST': ([512, 256], 128, 1, 10, 128),
    'EMNIST': ([512, 256], 128, 1, 10, 128),
    # 'EMNIST': ([512, 1024], 784, 1, 37, 128),
    'CIFAR-10':([512, 256], 128, 3, 10, 128),

    'CIFAR-100':([1024, 512], 256, 3, 100, 256)
}

GENERATORCONFIGS_DLG = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'FMNIST': ([512, 256], 588, 1, 10, 128),
    'EMNIST': ([512, 256], 588, 1, 10, 128),
    'CIFAR-10':([512, 256], 768, 3, 10, 128)
}



def model_pull(args, g_classifer=False, l_classifer=False, discriminator=False):
    if args.model_type == 'Lenet':  # 适用于：CIFAR-10
        if g_classifer == True:
            return Lenet(args).to(args.device), Lenet(args, classifier_or_not=g_classifer).to(args.device)

        elif l_classifer == True:
            return [], Lenet(args, classifier_or_not=l_classifer).to(args.device)

        elif discriminator == True:
            return [], Lenet(args, discriminator=discriminator).to(args.device)

        else:
            return Lenet(args).to(args.device), []

    elif args.model_type == 'ResNet_18':  # 适用于：MNIST, CIFAR-10, CIFAR-100
        if g_classifer == True:
            return ResNet(BasicBlock, [2, 2, 2, 2], args).to(args.device), Lenet(args, classifier_or_not=g_classifer).to(args.device)

        elif l_classifer == True:
            return [], Lenet(args, classifier_or_not=l_classifer).to(args.device)

        elif discriminator == True:
            return ResNet(BasicBlock, [2, 2, 2, 2], args).to(args.device), Lenet(args, discriminator=discriminator).to(args.device)

        else:
            return ResNet(BasicBlock, [2, 2, 2, 2], args).to(args.device), []

    elif args.model_type == 'ResNet_20':  # 适用于：MNIST, CIFAR-10, CIFAR-100
        if g_classifer == True:
            return ResNet(BasicBlock, [2, 3, 2, 2], args).to(args.device), Lenet(args, classifier_or_not=g_classifer).to(args.device)

        elif l_classifer == True:
            return [], Lenet(args, classifier_or_not=l_classifer).to(args.device)

        elif discriminator == True:
            return ResNet(BasicBlock, [2, 3, 2, 2], args).to(args.device), Lenet(args, discriminator=discriminator).to(args.device)

        else:
            return ResNet(BasicBlock, [2, 3, 2, 2], args).to(args.device), []

    else:
        raise Exception(f"unkonw model_type: {args.model_type}")
    # elif args.model_type == 'Bert':
    #     self.model = Bert(args)
    # ---------------------------------------


def generator_model_pull(args):

    # ------------ 凸模型 -----------
    if args.generator_model_type == 'generator':  # 适用于：MNIST, CIFAR-10, CIFAR-100, COVERTYPE, A9A, W8A
        return Generate(args)

def generator_model_pull_DLG(args):
    # ------------ 凸模型 -----------
    if args.generator_model_type == 'generator':  # 适用于：MNIST, CIFAR-10, CIFAR-100, COVERTYPE, A9A, W8A
        return Generate_DLG(args)


####DLG
def model_pull_DLG(args):
    if args.model_type == 'Lenet':  # 适用于：CIFAR-10
        return LeNet(args.data_set)

class LeNet(nn.Module):
    def __init__(self, data_set):
        super(LeNet, self).__init__()

        if data_set == 'CIFAR-10':
            input_dim = 3
            latent_dim = 768
        elif data_set == 'FMNIST':
            input_dim = 1
            latent_dim = 588
        elif data_set == 'EMNIST':
            input_dim = 1
            latent_dim = 588

        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(input_dim, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 10)
        )

    def weights_init(self, m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x, feature_or_not=False):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        if feature_or_not == True:
            out1 = out
        # print(out.size())
        out = self.fc(out)

        if feature_or_not == True:
            return out, out1
        else:
            return out


#####################FedGen方法的模型##############
##MNIST+Lenet_5  这里是简单的卷积层加线性层(暂定处理MNIST, CIFAR10, CIFAR100) 利用源代码给出的格式
class Lenet(nn.Module):
    def __init__(self, args, classifier_or_not=False, discriminator=False):
        super(Lenet, self).__init__()
        # define network layers
        dataset = args.data_set
        self.discriminator = discriminator
        print("Creating model for {}".format(dataset))
        self.dataset = dataset
        if classifier_or_not==True:
            self.cnn_configs, self.fc_configs, input_channel, self.output_dim, self.latent_dim = CLASSIFERCONFIGS[dataset]
        elif self.discriminator == True:
            self.cnn_configs, self.fc_configs, input_channel, self.output_dim, self.latent_dim = DISCRIMINATORCONFIGS[dataset]
        else:
            self.cnn_configs, self.fc_configs, input_channel, self.output_dim, self.latent_dim = CONFIGS_[dataset]

        print('Network configs:', self.cnn_configs+['||']+self.fc_configs)
        self.named_layers, self.layers, self.layer_names, self.named_parameter_layers =self.build_network(
            self.cnn_configs, self.fc_configs, input_channel, self.output_dim)
        self.n_parameters = len(list(self.parameters()))
        self.n_share_parameters = len(self.get_encoder())

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def weights_init(self, m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def build_network(self, cnn_configs, fc_configs, input_channel, output_dim):
        named_parameter_layers = []
        named_layers = {}
        layer_names = []
        kernel_size, stride, padding = 3, 2, 1
        # layers = nn.ModuleList()
        layers = []
        for i, x in enumerate(cnn_configs):
            if x == 'F':
                layer_name='flatten{}'.format(i)
                layer=nn.Flatten(1)
                layers = layers + [layer]
                layer_names = layer_names + [layer_name]
            elif x == 'M':
                pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                layer_name = 'pool{}'.format(i)
                layers = layers + [pool_layer]
                layer_names = layer_names + [layer_name]
            else:
                cnn_name = 'encode_cnn{}'.format(i)
                cnn_layer = nn.Conv2d(input_channel, x, stride=stride, kernel_size=kernel_size, padding=padding)
                named_layers[cnn_name] = [cnn_layer.weight, cnn_layer.bias]
                named_parameter_layers = named_parameter_layers + [cnn_name, cnn_name]

                bn_name = 'encode_batchnorm{}'.format(i)
                bn_layer = nn.BatchNorm2d(x)
                named_layers[bn_name] = [bn_layer.weight, bn_layer.bias]
                named_parameter_layers = named_parameter_layers + [bn_name, bn_name]

                relu_name = 'relu{}'.format(i)
                relu_layer = nn.ReLU()# no parameters to learn inplace=True

                layers = layers + [cnn_layer, bn_layer, relu_layer]
                layer_names = layer_names + [cnn_name, bn_name, relu_name]
                input_channel = x

        # finally, classification layer
        for i, x in enumerate(fc_configs):

            if x == 'L':
                relu_name = 'relu{}'.format(i)
                relu_layer = nn.ReLU()  # no parameters to learn inplace=True
                layers = layers + [relu_layer]
                layer_names = layer_names + [relu_name]
            else:
                fc_layer_name = 'fc{}'.format(i+1)
                fc_layer = nn.Linear(self.latent_dim, x)
                layers = layers + [fc_layer]
                layer_names = layer_names + [fc_layer_name]
                named_layers[fc_layer_name] = [fc_layer.weight, fc_layer.bias]
                named_parameter_layers = named_parameter_layers + [fc_layer_name, fc_layer_name]
                self.latent_dim = x

        fc_layer_name = 'fc{}'.format(len(fc_configs)+1)
        fc_layer = nn.Linear(self.latent_dim, output_dim)
        layers = layers + [fc_layer]
        layer_names = layer_names + [fc_layer_name]
        named_layers[fc_layer_name] = [fc_layer.weight, fc_layer.bias]
        named_parameter_layers = named_parameter_layers + [fc_layer_name, fc_layer_name]
        if self.discriminator:
            sigmod_name = 'sigmod{}'.format(i)
            sigmod_layer = nn.Sigmoid()  # no parameters to learn
            layers = layers + [sigmod_layer]
            layer_names = layer_names + [sigmod_name]

        return named_layers, nn.ModuleList(layers), layer_names, named_parameter_layers


    def get_parameters_by_keyword(self, keyword='encode'):
        params=[]
        for name, layer in zip(self.layer_names, self.layers):
            if keyword in name:
                #layer = self.layers[name]
                params += [layer.weight, layer.bias]
        return params

    def get_encoder(self):
        return self.get_parameters_by_keyword("encode")

    def get_decoder(self):
        return self.get_parameters_by_keyword("fc")

    def get_shared_parameters(self, detach=False):
        return self.get_parameters_by_keyword("fc")

    def get_learnable_params(self):
        return self.get_encoder() + self.get_decoder()

    def forward(self, x, start_layer_idx = 0, logit=False, latent_feature=False):
        """
        :param x:
        :param logit: return logit vector before the last softmax layer
        :param start_layer_idx: if 0, conduct normal forward; otherwise, forward from the last few layers (see mapping function)
        :return:
        """
        if start_layer_idx == None: #
            return self.mapping(x, logit=logit, latent_feature=latent_feature)
        elif start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit, latent_feature=latent_feature)

        results={}
        latent_features=[]
        z = x
        for idx in range(start_layer_idx, len(self.layers)):
            layer_name = self.layer_names[idx]
            if latent_feature and 'fc' in layer_name:
                latent_features.append(z)
            layer = self.layers[idx]
            z = layer(z)

        results['latent_features'] = latent_features

        if self.output_dim > 1:
            results['output'] = F.log_softmax(z, dim=1)
        else:
            results['output'] = z
        if logit:
            results['logit'] = z
        return results

    def mapping(self, z_input, start_layer_idx=None, logit=True, latent_feature=False):
        z = z_input
        n_layers = len(self.layers)
        if start_layer_idx == None:
            start_layer_idx = -len(self.fc_configs) - 1
        latent_features = []
        for layer_idx in range(n_layers + start_layer_idx, n_layers):
            layer = self.layers[layer_idx]
            if latent_feature:
                latent_features.append(z)
            z = layer(z)
        result={'latent_features' : latent_features}
        if self.output_dim > 1:
            out=F.log_softmax(z, dim=1)
        result['output'] = out
        if logit:
            result['logit'] = z
        return result



'''非凸模型：VGG-11 cifor-100 确定'''
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        class_num = 100
        self.feature = self.vgg_stack((1, 1, 2, 2, 2),
                                      ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)),
                                      (0.5, 0.5, 0.5, 0.5, 0.5))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            # nn.BatchNorm1d(4096, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, class_num)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def vgg_block(self, num_convs, in_channels, out_channels, dropout_p):
        net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
               nn.BatchNorm2d(out_channels, affine=True),
               nn.ReLU(inplace=True)]

        for i in range(num_convs - 1):
            net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            net.append(nn.BatchNorm2d(out_channels, affine=True))
            net.append(nn.ReLU(inplace=True))
        net.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*net)

    def vgg_stack(self, num_convs, channels, dropout_ps):
        net = []
        for n, c, d in zip(num_convs, channels, dropout_ps):
            in_c = c[0]
            out_c = c[1]
            net.append(self.vgg_block(n, in_c, out_c, d))
        return nn.Sequential(*net)



'''非凸模型：ResNet-18/34'''
##########################################
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, args):
        super().__init__()
        if args.data_set == 'CIFAR-100':
            self.num_classes = 100
        elif args.data_set == 'CIFAR-10':
            self.num_classes = 10

        self.in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 32, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 64, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 128, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 256, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256 * block.expansion, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, start_layer_idx = 0, logit=False, latent_feature=False):
        results = {}
        latent_features = []
        if start_layer_idx == 0:
            output = self.conv1(x)
            output = self.conv2_x(output)
            output = self.conv3_x(output)
            output = self.conv4_x(output)
            output = self.conv5_x(output)
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
            if latent_feature:
                latent_features.append(output)
            output = self.fc1(output)
            if latent_feature:
                latent_features.append(output)
            output = self.fc2(output)

        elif start_layer_idx == None:
            if latent_feature:
                latent_features.append(x)
            output = self.fc1(x)
            if latent_feature:
                latent_features.append(output)
            output = self.fc2(output)

        results['latent_features'] = latent_features

        if self.num_classes > 1:
            results['output'] = F.log_softmax(output, dim=1)
        else:
            results['output'] = output

        if logit:
            results['logit'] = output

        return results

# def resnet18():
#     """ return a ResNet 18 object
#     """
#     return ResNet(BasicBlock, [2, 2, 2, 2])

# def resnet34():
#     """ return a ResNet 34 object
#     """
#     return ResNet(BasicBlock, [3, 4, 6, 3])


###########生成器########## 生成器的格式给定，可根据预先定义的网络参数给定
class Generate(nn.Module):
    def __init__(self, args):
        super(Generate, self).__init__()
        dataset = args.data_set
        print("Dataset {}".format(dataset))
        self.embedding = args.embedding
        self.dataset = dataset
        self.device = args.device
        #self.model=model
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = GENERATORCONFIGS[dataset]
        input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        self.fc_configs = [input_dim] + self.hidden_dim
        # self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    # def init_loss_fn(self):
    #     self.crossentropy_loss=nn.NLLLoss(reduce=False).to(self.device) # same as above
    #     self.diversity_loss = DiversityLoss(metric='l2').to(self.device) ##这里可以进行修改
    #     self.dist_loss = nn.MSELoss().to(self.device)

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        ### FC modules ####
        # self.fc_layers = nn.ModuleList()
        fc_layers = []
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            fc_layers = fc_layers + [fc, bn, act]
        self.fc_layers = nn.ModuleList(fc_layers)
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, eps=None, verbose=True, eps_=False):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        result = {}
        batch_size = labels.shape[0]
        if eps_==False:
            eps = torch.rand((batch_size, self.noise_dim)) # sampling from Gaussian
        if verbose:
            result['eps'] = eps
        if self.embedding: # embedded dense vector
            y_input = self.embedding_layer(labels)
        else: # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class)
            y_input.zero_()
            #labels = labels.view
            y_input.scatter_(1, labels.view(-1,1), 1)
        result['y_input'] = y_input
        z = torch.cat((eps, y_input), dim=1)
        result['eps_y'] = z
        ### FC layers
        z = z.to(self.device)
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std

class Generate_DLG(nn.Module):
    def __init__(self, args):
        super(Generate_DLG, self).__init__()
        dataset = args.data_set
        print("Dataset {}".format(dataset))
        self.embedding = args.embedding
        self.dataset = dataset
        self.device = args.device
        #self.model=model
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = GENERATORCONFIGS_DLG[dataset]
        input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        self.fc_configs = [input_dim] + self.hidden_dim
        # self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    # def init_loss_fn(self):
    #     self.crossentropy_loss=nn.NLLLoss(reduce=False).to(self.device) # same as above
    #     self.diversity_loss = DiversityLoss(metric='l2').to(self.device) ##这里可以进行修改
    #     self.dist_loss = nn.MSELoss().to(self.device)

    def build_network(self):
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        ### FC modules ####
        # self.fc_layers = nn.ModuleList()
        fc_layers = []
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            fc_layers = fc_layers + [fc, bn, act]
        self.fc_layers = nn.ModuleList(fc_layers)
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, eps=None, verbose=True, eps_=False):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        result = {}
        batch_size = labels.shape[0]
        if eps_==False:
            eps = torch.rand((batch_size, self.noise_dim)) # sampling from Gaussian
        if verbose:
            result['eps'] = eps
        if self.embedding: # embedded dense vector
            y_input = self.embedding_layer(labels)
        else: # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class)
            y_input.zero_()
            #labels = labels.view
            y_input.scatter_(1, labels.view(-1,1), 1)
        result['y_input'] = y_input
        z = torch.cat((eps, y_input), dim=1)
        result['eps_y'] = z
        ### FC layers
        z = z.to(self.device)
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std


