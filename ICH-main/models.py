import torch
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck

model_urls = {
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}

def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    model.load_state_dict(state_dict)
    return model

def resnext101_32x8d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

class ResNeXtModel(torch.nn.Module):
    def __init__(self):
        super(ResNeXtModel, self).__init__()
        resnext = resnext101_32x8d_wsl()
        self.base = torch.nn.Sequential(*list(resnext.children())[:-1])
        self.fc = torch.nn.Sequential(torch.nn.Linear(2048,6))
    
    def forward(self, input):
        features = self.base(input).reshape(-1, 2048)
        out = self.fc(features)
        return out, features
