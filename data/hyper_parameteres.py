# hyper_parameters

hyper_parameters = {}
hyper_parameters['image_size'] = [384, 384]
hyper_parameters['batch_size'] = 7
hyper_parameters['model_version'] = 1
hyper_parameters['num_classes'] = 1601
hyper_parameters['vocab_dim'] = 30525
hyper_parameters['features_number'] = 10
hyper_parameters['confidence_threshold'] = 0.1
hyper_parameters['max_len'] = 10
hyper_parameters['topk'] = 5
hyper_parameters['patience'] = 6
hyper_parameters['shrink_factor'] = 0.8

## version 1 : 'efficientdet_d1-4c7ebaf2.pth'
## version 2 : 'efficientdet_d2-cb4ce77d.pth'
## version 3 : 'efficientdet_d3-b0ea2cbc.pth'
## version 4 : 'efficientdet_d4-5b370b7a.pth'
## version 5 : 'efficientdet_d5-ef44aea8.pth'
hyper_parameters['pretrained_version'] = 'efficientdet_d1-4c7ebaf2.pth'

def collate_fn(batch):
  return tuple(zip(*batch))
