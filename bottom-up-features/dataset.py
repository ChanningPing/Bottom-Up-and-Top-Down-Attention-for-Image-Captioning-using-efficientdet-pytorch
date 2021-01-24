# effdet_dataset

class effdet_dataset(torch.utils.data.Dataset):
  def __init__(self, path, attributes, transform):
    self.path = path
    self.attributes = attributes
    self.transform = transform

  def __len__(self):
    return len(self.attributes)

  def __getitem__(self, index):
    attribute = self.attributes.loc[index]
    
    image_id = attribute['image_id']
    direc_id = attribute['directory'] + 1
    file_id = self.path + f'{direc_id}/' + f'{image_id}.jpg'

    image = cv2.imread(file_id, cv2.IMREAD_COLOR).copy().astype(np.uint8)#float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)#float32)
    h, w, _ = image.shape

    boxes = np.array(attribute['box'])
    boxes[:, 0] = np.clip(boxes[:, 0], a_min = 0, a_max = w)
    boxes[:, 1] = np.clip(boxes[:, 1], a_min = 0, a_max = h)
    boxes[:, 2] = np.clip(boxes[:, 0] + boxes[:, 2], a_min = 0, a_max = w)
    boxes[:, 3] = np.clip(boxes[:, 1] + boxes[:, 3], a_min = 0, a_max = h)
    boxes = self.__make_box_form__(boxes, h, w)

    labels = torch.Tensor(attribute['class']).long()

    target = {}
    target['boxes'] = boxes
    target['labels'] = labels

    if self.transform:
      for i in range(10):
        sample = self.transform(**{
            'image': image,
            'bboxes': target['boxes'],
            'labels': target['labels']})
        if len(sample['bboxes']) > 0:
          image = sample['image']
          target['boxes'] = torch.stack(tuple(map(torch.Tensor, zip(*sample['bboxes'])))).permute(1, 0)
          target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]
          break

    return image, target, image_id

  def __make_box_form__(self, boxes, h, w):
    boxes[:, 0] = np.clip(boxes[:, 0] - 0.01, a_min = 0, a_max = w)
    boxes[:, 1] = np.clip(boxes[:, 1] - 0.01, a_min = 0, a_max = h)
    boxes[:, 2] = np.clip(boxes[:, 2] + 0.01, a_min = 0, a_max = w)
    boxes[:, 3] = np.clip(boxes[:, 3] + 0.01, a_min = 0, a_max = h)

    return boxes
        
train_dataset = effdet_dataset(path = path, 
                               attributes = train_attributes, 
                               transform = get_train_transforms())

valid_dataset = effdet_dataset(path = path,
                               attributes = valid_attributes,
                               transform = get_valid_transforms())
