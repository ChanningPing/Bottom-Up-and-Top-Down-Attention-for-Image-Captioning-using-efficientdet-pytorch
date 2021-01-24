# upload image 

import cv2
import numpy as np
from tqdm import tqdm
import torch

for i in range(55):
  os.mkdir('/content/gdrive/My Drive/top-down-bottom-up/visual_genome_dataset/vg_part' + f'{i+1}')

class uploade_image_faster_dataset(torch.utils.data.Dataset):
  def __init__(self, attributes):
    self.attributes = attributes

  def __len__(self):
    return len(self.attributes)

  def __getitem__(self, index):
    attribute = self.attributes.loc[index]
    image_id = attribute['image_id']
    if attribute['one_or_two'] == 1:
      image = cv2.imread('./VG_100K/' + f'{image_id}.jpg', cv2.IMREAD_COLOR).copy().astype(np.float32)
    else:
      image = cv2.imread('./VG_100K_2/' + f'{image_id}.jpg', cv2.IMREAD_COLOR).copy().astype(np.float32)
    direc = attribute['directory'] + 1
    return direc, image_id, image

uploade_image_faster_dataset = uploade_image_faster_dataset(attributes)

uploade_image_faster_dataloader = torch.utils.data.DataLoader(
    uploade_image_faster_dataset,
    batch_size = 1,
    pin_memory = False,
    drop_last = False,
    shuffle = False,
    num_workers = 1)

uploade_image_faster_book = tqdm(uploade_image_faster_dataloader, 
                                 total = len(uploade_image_faster_dataloader))

for step, data in enumerate(uploade_image_faster_book):
  direc, image_id, image = data
  _ = cv2.imwrite('/content/gdrive/My Drive/top-down-bottom-up/visual_genome_dataset/vg_part' 
                  + f'{direc.tolist()[0]}/' + f'{image_id.tolist()[0]}.jpg', np.array(image[0]))
