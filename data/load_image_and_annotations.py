# prepare visual genome dataset : attributes, captions

import json
import pandas as pd

path = '/content/gdrive/My Drive/top-down-bottom-up/visual_genome_dataset/vg_part'

!wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
!unzip images.zip

!wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
!unzip images2.zip

with open('/content/gdrive/My Drive/Visual_Genome_Annotations/attributes.json') as json_file:
  attributes = json.load(json_file)
attributes = pd.DataFrame(attributes)

with open('/content/gdrive/My Drive/Visual_Genome_Annotations/captions.json') as json_file:
  captions = json.load(json_file)
captions = pd.DataFrame(captions)
