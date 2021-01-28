# resnet_encoder

from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn.functional as F

resnet = ptcv_get_model('resnet50', pretrained = True)

class encoder(nn.Module):
  def __init__(self, resnet, effdet, predict_function, fea_num, conf_thres):
    super(encoder, self).__init__()
    self.backbone = resnet.features
    self.effdet = effdet
    self.predict_function = predict_function
    self.fea_num = fea_num
    self.conf_thres = conf_thres
    self.patch_size = 196
    self.dropout = nn.Dropout(p = 0.0)

    self.effdet.eval()
  
  def __getpatch__(self, images):
    batch_boxes = self.predict_function(self.effdet, images, self.conf_thres)[0] 

    batch = images.shape[0]

    batch_patches = []
    for i in range(batch):
      image = images[i] 
      boxes = batch_boxes[i].astype(np.int32) 

      patches = []
      for j in range(len(boxes)):
        box = boxes[j]
        img_seg = image[:, box[1]:box[3], box[0]:box[2]]
        patch = image * 0
        patch[:, box[1]:box[3], box[0]:box[2]] = img_seg
        if img_seg.shape[1] * img_seg.shape[2] != 0:
          patch = F.adaptive_avg_pool2d(patch, self.patch_size)  
          patches.append(patch)
      if len(patches) == 0:
        patches.append(F.adaptive_avg_pool2d(image, self.patch_size))

      if len(patches) <= self.fea_num:
        patches += [patches[-1]] * (self.fea_num - len(patches))
      patches = torch.stack(patches) # patches size : (features_num, 3, patch_size, patch_size)

      batch_patches.append(patches[:self.fea_num, :, :, :])
    return batch_patches

  def forward(self, images):
    batch = images.shape[0]
    batch_patches = self.__getpatch__(images) # (N, features_num, 3, patch_size, patch_size)
    
    batch_features = []
    with torch.no_grad():
      for patches in batch_patches:
        features = self.backbone(patches)
        ## features = self.dropout(features)
        batch_features.append(features)

    return torch.stack(batch_features).reshape(batch, self.fea_num, -1)

resnet_encoder = encoder(resnet = resnet, 
                         effdet = effdet, 
                         predict_function = effdet_predict, 
                         fea_num = hyper_parameters['features_number'],
                         conf_thres = hyper_parameters['confidence_threshold'])
       
  
  
  
  
  
        
# attend_module

from torch.nn.utils.weight_norm import weight_norm

class attend_module(nn.Module):
  def __init__(self, features_dim, lstm_dim, attention_dim, dropout = 0.5):
    super(attend_module, self).__init__()
    self.features_att = weight_norm(nn.Linear(features_dim, attention_dim, bias = False))
    self.decoder_att = weight_norm(nn.Linear(lstm_dim, attention_dim, bias = False))
    self.total_att = weight_norm(nn.Linear(attention_dim, 1, bias = False))  
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim = 1)
    self.dropout = nn.Dropout(p = dropout)

  def forward(self, batch_features, lstm_hidden):
    attend1 = self.features_att(batch_features) # (N, features_num, attention_dim)
    attend2 = self.decoder_att(lstm_hidden) # (N, attention_dim)
    attend = self.total_att(self.dropout(self.tanh(attend1 + attend2.unsqueeze(1)))) # (N, features_num, 1)
    alpha = self.softmax(attend.squeeze(2)) # (N, features_num)
    
    batch_weighted_features = torch.sum(batch_features * alpha.unsqueeze(2), dim = 1)   
    
    return batch_weighted_features # (N, features_dim)

attend_net = attend_module(features_dim = 2048,
                           lstm_dim = 1024,
                           attention_dim = 1024)
                           


  
  
  
  
  
# gru_decoder : bottom-up-top-down

class decoder(nn.Module):
  def __init__(self, encoder, attend_module, features_dim, gru_dim, embed_dim, dropout = 0.5):
    super(decoder, self).__init__()
    self.encoder = encoder
    self.attend_module = attend_module
    self.features_dim = features_dim
    self.gru_dim = gru_dim
    self.embed_dim = embed_dim
    self.vocab_dim = hyper_parameters['vocab_dim']
    self.seq_len = hyper_parameters['max_len']

    self.embed_layer = nn.Embedding(self.vocab_dim, self.embed_dim)
    self.top_down_attention_lstm = nn.GRUCell(self.gru_dim + self.features_dim + self.embed_dim,
                                              self.gru_dim)
    self.language_lstm = nn.GRUCell(self.gru_dim + self.features_dim,
                                    self.gru_dim)
    self.fc_layer = weight_norm(nn.Linear(self.gru_dim, self.vocab_dim))
    self.softmax = nn.Softmax(dim = 1)
    self.dropout = nn.Dropout(p = dropout)
    
    self.__init_weights__()
  
  def __init_weights__(self):
    ## self.embed_layer.weight.data.uniform_(- 0.1, 0.1)
    self.fc_layer.bias.data.fill_(0)
    self.fc_layer.weight.data.uniform_(- 0.1, 0.1)

  def __init_gru_state__(self, batch):
    h = torch.zeros(batch, self.gru_dim).to(device)
    return h 
  
  def __random_topk__(self, pred, k): 
    prob_distribution = self.softmax(pred)
    top_indices = prob_distribution.topk(k = k).indices.squeeze(0)
    return random.choice(top_indices).unsqueeze(0)

  def forward(self, images, input_ids = None):
    batch = images.shape[0] # (N)
    batch_features = self.encoder(images)

    input_embed = self.embed_layer(torch.Tensor(batch * [30523]).to(device).long())      
    mean_pooled_features = batch_features.mean(1) # (N, features_dim)

    h1 = self.__init_gru_state__(batch) # (N, gru_dim)
    h2 = self.__init_gru_state__(batch) # (N, gru_dim)
    
    preds = [] 
    for step in range(self.seq_len):

      h1 = self.top_down_attention_lstm(
          torch.cat([h2, mean_pooled_features, input_embed], dim = 1), 
          h1)
      
      batch_weighted_features = self.attend_module(batch_features, h1)

      h2 = self.language_lstm(
          torch.cat([batch_weighted_features, h1], dim = 1), 
          h2)
      
      pred = self.fc_layer(self.dropout(h2)) # (N, vocab_dim)
      preds.append(pred.unsqueeze(1))
       
      if (input_ids is not None) & (step != self.seq_len - 1): # train & valid
        input_embed = self.embed_layer(input_ids[:, step + 1]) 
      else: # inference
        input_embed = self.embed_layer(self.__random_topk__(pred = pred, k = 3))

    return torch.cat(preds, dim = 1)

gru_decoder = decoder(encoder = resnet_encoder,
                      attend_module = attend_net,
                      features_dim = 2048,
                      gru_dim = 1024,
                      embed_dim = 1024)
