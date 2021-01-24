import os
import pickle

def get_cls(input):
  cls_output = []
  for i in range(len(objects)):
    if input in objects[i]:
      cls_output.append(i)

  if len(cls_output) == 0:
    cls_output.append(1600)
  return cls_output[0]

all_direcs = []
all_ids = []
all_boxes = []
all_classes = []
one_or_two = []
for i in range(len(attributes)):
  if i % 10000 == 0:
    print(i)
  image_id = captions.loc[i]['id']
  attribute = attributes.loc[i]['attributes']

  if os.path.exists('./VG_100K/' + f'{image_id}.jpg'):
    one_or_two.append(1)
  else:
    one_or_two.append(2)

  classes = []
  boxes = []
  for j in range(len(attribute)):
    cls = get_cls(attribute[j]['names'][0]) 
    box = [attribute[j]['x'], attribute[j]['y'], attribute[j]['w'], attribute[j]['h']]
    classes.append(cls)
    boxes.append(box)

  if len(boxes) == 0:
    boxes = [[0, 0, 0.01, 0.01]]
    classes = [1600]
  
  direc = i // 2000
  
  all_direcs.append(direc)
  all_ids.append(image_id)
  all_boxes.append(boxes)
  all_classes.append(classes)

attributes['box'] = all_boxes
attributes['class'] = all_classes
attributes['image_id'] = all_ids
attributes['directory'] = all_direcs
attributes['one_or_two'] = one_or_two

attributes = attributes.to_dict()

with open('/content/gdrive/My Drive/attributes.pickle','wb') as fw:
  pickle.dump(attributes, fw)

phrases = []
all_direcs = []
for i in range(len(captions)):
  if i % 10000 == 0:
    print(i)
  phrase = []
  for j in range(3):
    phrase.append(captions.loc[i]['regions'][j]['phrase'])
  phrase = ', '.join(phrase)
  phrases.append(phrase)

  direc = i // 2000
  all_direcs.append(direc)
captions['phrases'] = phrases
captions['directory'] = all_direcs

captions = captions.to_dict()

with open('/content/gdrive/My Drive/captions.pickle','wb') as fw:
  pickle.dump(captions, fw)

with open('/content/gdrive/My Drive/attributes.pickle', 'rb') as fr:
  attributes = pickle.load(fr)

attributes = pd.DataFrame(attributes)

with open('/content/gdrive/My Drive/captions.pickle', 'rb') as fr:
  captions = pickle.load(fr)

captions = pd.DataFrame(captions)
