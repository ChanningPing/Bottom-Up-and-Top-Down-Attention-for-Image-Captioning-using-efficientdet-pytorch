# effdet_model

device = 'cuda'

from effdet import get_efficientdet_config, efficientdet
from effdet import DetBenchTrain
from effdet.efficientdet import HeadNet
import torch.nn as nn

def freeze_bn(self):
  for m in self.modules():
    if isinstance(m, nn.BatchNorm2d):
      m.eval()

def get_net():
  model_version = hyper_parameters['model_version']
  pretrained_version = hyper_parameters['pretrained_version']
  config = get_efficientdet_config('tf_efficientdet_d' + f'{model_version}')
  config.image_size = hyper_parameters['image_size']
  net = efficientdet.EfficientDet(config, pretrained_backbone = True)
  checkpoint = torch.load('/content/gdrive/My Drive/Impact_Detection/pretrained_efficientdet/' +
                          f'{pretrained_version}')
  net.load_state_dict(checkpoint)
  net.reset_head(num_classes = hyper_parameters['num_classes'])
  net.class_net = HeadNet(config, num_outputs = config.num_classes)
  return DetBenchTrain(net, config)

net = get_net()
net = net.apply(freeze_bn)
