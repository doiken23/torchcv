import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, pretrained=True, skip=False, return_indices=False):
        super(VGG16, self).__init__()
        self.skip = skip
        self.return_indices = return_indices
        vgg16 = models.vgg16(pretrained)
        features = vgg16.features
        self.conv1_1 = features[0]
        self.conv1_2 = features[2]
        self.conv2_1 = features[5]
        self.conv2_2 = features[7]
        self.conv3_1 = features[10]
        self.conv3_2 = features[12]
        self.conv3_3 = features[14]
        self.conv4_1 = features[17]
        self.conv4_2 = features[19]
        self.conv4_3 = features[21]
        self.conv5_1 = features[24]
        self.conv5_2 = features[26]
        self.conv5_3 = features[28]

    def forward(self, x):
        if self.skip:
            features = []
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        if self.skip:
            features.append(h)
        h, idx1 = F.max_pool2d(h, 2, return_indices=True)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        if self.skip:
            features.append(h)
        h, idx2 = F.max_pool2d(h, 2, return_indices=True)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        if self.skip:
            features.append(h)
        h, idx3 = F.max_pool2d(h, 2, return_indices=True)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        if self.skip:
            features.append(h)
        h, idx4 = F.max_pool2d(h, 2, return_indices=True)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        if self.skip:
            features.append(h)
        h, idx5 = F.max_pool2d(h, 2, return_indices=True)

        out = [h]
        if self.skip:
            out.append(features)
        if self.return_indices:
            out.append([idx1, idx2, idx3, idx4, idx5])
        if not (self.skip or self.return_indices):
            return h
        else:
            return out

def vgg16(pretrained, **kwargs):
    return VGG16(pretrained, **kwargs)
