# caption dataset

class caption_dataset(torch.utils.data.Dataset):
  def __init__(self, captions, path, tokenizer, max_len, mode):
    self.captions = captions
    self.path = path
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.mode = mode
    self.start = self.tokenizer.convert_tokens_to_ids('[start]')
    self.end = self.tokenizer.convert_tokens_to_ids('[end]')
    self.pad = self.tokenizer.convert_tokens_to_ids('[pad]')

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, index):

    caption = self.captions.loc[index]

    image_id = caption['id']
    direc_id = caption['directory'] + 1
    file_id = self.path + f'{direc_id}/' + f'{image_id}.jpg'

    image = cv2.imread(file_id, cv2.IMREAD_COLOR).copy().astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) 
    if self.mode == 'train':
      image = self.__train_imgtransform__()(image = image)['image']
    
    else:
      image = self.__valid_imgtransform__()(image = image)['image']

    phrases = caption['phrases']

    input_id = self.tokenizer.encode(phrases)[1:-1]
    target = input_id + [self.end]
    input_id = [self.start] + input_id

    input_id = self.__pad__(input_id, self.pad, self.max_len - len(input_id))
    input_id = torch.Tensor(input_id)[:self.max_len].long()

    target = self.__pad__(target, self.pad, self.max_len - len(target))
    target = torch.Tensor(target)[:self.max_len].long()
    
    return image, input_id, target

  def __pad__(self, input, pad_val, pad_len):
    return input + [pad_val] * pad_len
  
  def __train_imgtransform__(self):
    return A.Compose(
        [#A.OneOf([A.HueSaturationValue(hue_shift_limit = 0.2, sat_shift_limit = 0.2, 
         #                            val_shift_limit = 0.2, p = 0.9),
         #       A.RandomBrightnessContrast(brightness_limit = 0.2, 
         #                                  contrast_limit = 0.2, p = 0.9),], p = 0.9),
         A.ToGray(p = 0.01),         
         A.HorizontalFlip(p = 0.5),
         A.VerticalFlip(p = 0.5),
         A.RandomRotate90(p = 0.5),
         A.Transpose(p = 0.5),
         #A.JpegCompression(quality_lower = 85, quality_upper = 95, p = 0.2),
         #A.OneOf([A.Blur(blur_limit = 3, p = 1.0),
         #         A.MedianBlur(blur_limit = 3, p = 1.0)], p = 0.1),
         A.Resize(height = 384, width = 384, p = 1.0),
         A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
         A.Cutout(num_holes = 8, max_h_size = 16, max_w_size = 16, fill_value = 0, p = 0.5),
         ToTensorV2(p = 1.0)],
         p = 1.0)

  def __valid_imgtransform__(self):
    return A.Compose(
        [A.Resize(height = 384, width = 384, p = 1.0),
        A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
        ToTensorV2(p = 1.0)], 
        p = 1.0)
    
train_dataset = caption_dataset(captions = train_captions, 
                                path = path, 
                                tokenizer = bert_tokenizer, 
                                max_len = hyper_parameters['max_len'],
                                mode = 'train')

valid_dataset = caption_dataset(captions = valid_captions, 
                                path = path, 
                                tokenizer = bert_tokenizer, 
                                max_len = hyper_parameters['max_len'],
                                mode = 'valid')
