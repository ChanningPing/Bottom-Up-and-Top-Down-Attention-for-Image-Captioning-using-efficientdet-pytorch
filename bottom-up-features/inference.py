# predict function

from effdet import DetBenchPredict

def effdet_predict(model, images, score_threshold):
  images = images.to(device)

  box_list = []
  score_list = []
  label_list = []
  with torch.no_grad():
    detections = model(images)
    batch = images.shape[0]
    for i in range(batch):

      boxes = detections[i].detach().cpu().numpy()[:, :4]
      scores = detections[i].detach().cpu().numpy()[:, 4]
      label = detections[i].detach().cpu().numpy()[:, 5]
     
      indices = np.where(scores >= score_threshold)[0]
      box_list.append(boxes[indices])
      score_list.append(scores[indices])
      label_list.append(label[indices])
  
  return box_list, score_list, label_list
  
  
  
# inference function

def effdet_inference(model, 
                     predict_function, 
                     dataloader, 
                     score_threshold,
                     resized_image_size,
                     orig_image_size):
  count = 0
  for images, targets, _ in dataloader:
    images = torch.stack(images).to(device)
    box_list, score_list, _ = predict_function(model, images, score_threshold)

    batch = len(images)
    for i in range(batch):
      sample = images[i].permute(1, 2, 0)
      sample = sample.cpu().numpy()
      boxes = box_list[i].astype(np.int32)
      boxes = boxes.clip(min = 0, max = resized_image_size[0] - 1)
      scores = score_list[i]
      if len(scores) >= 1:
        fig, axis = plt.subplots(1, 1, figsize = (16, 8))
        sample = cv2.resize(sample , (int(orig_image_size[0]), int(orig_image_size[1])))
        for box, score in zip(boxes, scores):
          box[0] = box[0] * orig_image_size[0] / resized_image_size[0] 
          box[1] = box[1] * orig_image_size[1] / resized_image_size[1]
          box[2] = box[2] * orig_image_size[0] / resized_image_size[0]
          box[3] = box[3] * orig_image_size[1] / resized_image_size[1]
          cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1.0, 0.0, 0.0), 3)
        axis.set_axis_off()
        axis.imshow(sample.astype(np.float32));
        count += 1
    if count >= 10:
      break
