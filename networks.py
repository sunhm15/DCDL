import torch
import torch.nn as nn 
import numpy as np
from torch.utils import model_zoo
from collections import OrderedDict

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def conv(ic, oc, k, s, p):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(oc),
    )
        
class CM_block(nn.Module):
    def __init__(self):
        super(CM_block, self).__init__()
        
    def forward(self, x, gamma, beta):
        # print(x.size(), gamma, beta)
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)+1.
        
        x = gamma * x + beta
        
        return x

class VGG_cmCore(nn.Module):
 
    def __init__(self, cfg, batch_norm=False, CM_flag=False, k=3):
        super(VGG_cmCore, self).__init__()
        self.cm = CM_block() if CM_flag else None
        self.models, self.models_cm = self.make_layers(cfg, batch_norm, CM_flag)
        self.k = k
 
    def make_layers(self, cfg, batch_norm, CM_flag):
        layers = []
        layers_cm = []
        in_channels = 3
        scale = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if CM_flag and scale>=4:
                    layers_cm += [nn.MaxPool2d(kernel_size=2, stride=2)]
                scale += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if not CM_flag:
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                else:
                    conv2d_cm = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                        if scale>=4:
                            layers_cm += [conv2d_cm, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                        if scale>=4:
                            layers_cm += [conv2d_cm, nn.ReLU(inplace=True)]
                in_channels = v
        
        return nn.ModuleList(layers), nn.ModuleList(layers_cm)
 
    def forward(self, x, gamma=None, beta=None):
        st_channels = 0
        scale = 0
        scale_batch = 13
        prescale_index = 0
        for j, m in enumerate(self.models):
            if gamma is None and self.cm and scale>=4:
                x = self.models_cm[j-prescale_index-1](x)
            else:
                x = m(x)
                if isinstance(m, nn.MaxPool2d):
                    scale += 1
                    prescale_index = j
                elif gamma is not None and self.cm and isinstance(m, nn.BatchNorm2d):
                    scale_batch -= 1
                    if scale_batch<self.k:
                        channels = x.size(1)
                        x = self.cm(x, gamma[:, st_channels:st_channels+channels], beta[:, st_channels:st_channels+channels])
                        st_channels += channels
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.001)
                nn.init.constant_(m.bias, 0)


# 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False, withtop = True):
        super(VGG, self).__init__()
        self.features = features
        self.withtop = withtop
        if self.features.cm:
            self.encoder_cm = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
            )
        self.classifer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()
 
    def forward(self, x, beta=None, gamma=None):
        x = self.features(x, beta, gamma)
        x = x.view(x.size(0), -1)
        if self.features.cm and gamma is None:
            return self.encoder_cm(x)
        else:
            return self.classifer(x)
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.001)
                nn.init.constant_(m.bias, 0)

def vgg16_dcdl(pretrained=False, load_wiki=None, k=3, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(VGG_cmCore(cfg['D'], batch_norm=True, CM_flag=True, k=k), **kwargs)
    if load_wiki:
        state_dict = torch.load(load_wiki)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "features" in k:
                num = int(k.split('.')[2])
                new_state_dict[k] = v
                if num>=34:
                    new_state_dict[k.replace(str(num), str(num-34)).replace("models", "models_cm")] = v
        model.load_state_dict(new_state_dict, strict=False)
        for param in model.features.models.parameters():
            param.requires_grad = False
            
    return model

def vgg16_bn(pretrained=False, load_wiki=None, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(VGG_cmCore(cfg['D'], batch_norm=True, CM_flag=False), **kwargs)
    
    if load_wiki:
        state_dict = torch.load(load_wiki)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "classifer" not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        for param in model.features.parameters():
            param.requires_grad = False
    elif pretrained:
        state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "features" in k:
                name = k.replace('features', 'models')
                new_state_dict[name] = v
        model.features.load_state_dict(new_state_dict)
        for param in model.features.parameters():
            param.requires_grad = False

    return model

class VGG_DCDL(nn.Module):
    def __init__(self, n_classes, n_classes_gender, n_classes_race, load_wiki=None, k=3):
        super(VGG_DCDL, self).__init__()

        n_channels = np.sum([c for c in cfg['D'][::-1] if c!='M'][:k])
        self.n_classes_gender = n_classes_gender
        self.n_classes_race = n_classes_race
        self.n_channels = n_channels

        self.predictor = vgg16_dcdl(num_classes=n_classes, pretrained=True, load_wiki=load_wiki, k=k)

        self.task_estimate = nn.Sequential(nn.Linear(4096, 2048),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(2048, n_classes_gender+n_classes_race))
        self.cm_generator = nn.Sequential(nn.Linear(4096, 2048),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(2048, 2*n_channels))

    def forward(self, x):
        batch_size = x.size(0)

        task_feature = self.predictor(x)
        task_class = self.task_estimate(task_feature)
        gender_result = task_class[:,:self.n_classes_gender]
        race_result = task_class[:,self.n_classes_gender:]
        cm_vector = self.cm_generator(task_feature).view(batch_size, 2, -1)

        x = self.predictor(x, cm_vector[:,0,:], cm_vector[:,1,:])
        
        if self.n_classes_race==0:
            return x, gender_result
        else:
            return x, gender_result, race_result


def make_model(model_dict):
    return VGG_DCDL(model_dict['n_classes'], model_dict['n_classes_gender'], model_dict['n_classes_race'], model_dict['load_wiki'], model_dict['k'])
