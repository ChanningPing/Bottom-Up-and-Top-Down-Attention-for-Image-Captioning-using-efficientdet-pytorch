# effdet_fitter

import warnings
import os
from datetime import datetime
import time
from tqdm import tqdm
from glob import glob

warnings.filterwarnings('ignore')

class effdet_fitter:
  def __init__(self, model, config):
    self.model = model
    self.config = config
    
    self.epoch = 0
    self.base_dir = f'/content/gdrive/My Drive/top-down-bottom-up/{self.config.folder}'
    if not os.path.exists(self.base_dir):
      os.makedirs(self.base_dir)
        
    self.log_path = f'{self.base_dir}/log.txt'
    self.best_summary_loss = 10 ** 5

    param_optimizer = list(self.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}] 

    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
    self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
    self.log(f'effdet_fitter prepared. device is {device}')
    
  def fit(self, train_dataloader, valid_dataloader):
    for e in range(self.config.n_epochs):
      if self.config.verbose:
        lr = self.optimizer.param_groups[0]['lr']
        timestamp = datetime.utcnow().isoformat()
        self.log(f'\n{timestamp}\nLR: {lr}')

      t = time.time()
      summary_loss = self.train_function(train_dataloader)

      self.log(f'[RESULT]: train. epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
      self.save(f'{self.base_dir}/last-checkpoint.bin')

      t = time.time()
      summary_loss = self.valid_function(valid_dataloader)

      self.log(f'[RESULT]: valid. epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
      if summary_loss.avg < self.best_summary_loss:
        self.best_summary_loss = summary_loss.avg
        self.model.eval()
        self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

        for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
          os.remove(path)

      if self.config.validation_scheduler:
        self.scheduler.step(metrics = summary_loss.avg)

      self.epoch += 1

  def valid_function(self, valid_dataloader):
    ## self.model.eval()
    summary_loss = averagemeter()
    t = time.time()
    valid_book = tqdm(valid_dataloader, total = len(valid_dataloader))
    for step, (images, targets, image_ids) in enumerate(valid_book):
      with torch.no_grad():
        images = torch.stack(images)
        batch_size = images.shape[0]
        images = images.to(device).float()
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]

        targets = {}
        targets['bbox'] = boxes
        targets['cls'] = labels

        loss = self.model(images, targets)
        summary_loss.update(loss['loss'].detach().item(), batch_size)

    return summary_loss

  def train_function(self, train_dataloader):
    self.model.train()
    summary_loss = averagemeter()
    t = time.time()
    train_book = tqdm(train_dataloader, total = len(train_dataloader))
    for step, (images, targets, image_ids) in enumerate(train_book):
      images = torch.stack(images)
      images = images.to(device).float()
      batch_size = images.shape[0]
      boxes = [target['boxes'].to(device).float() for target in targets]
      labels = [target['labels'].to(device).float() for target in targets]

      targets = {}
      targets['bbox'] = boxes
      targets['cls'] = labels

      self.optimizer.zero_grad()
      
      total_loss = self.model(images, targets)
      loss = total_loss['loss']
      
      loss.backward()

      summary_loss.update(loss.detach().item(), batch_size)

      self.optimizer.step()

      if self.config.step_scheduler:
          self.scheduler.step()

    return summary_loss
    
  def save(self, path):
    ## self.model.eval()
    torch.save({
        'model_state_dict': self.model.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'best_summary_loss': self.best_summary_loss,
        'epoch': self.epoch,
    }, path)

  def load(self, path):
    checkpoint = torch.load(path)
    self.model.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    self.best_summary_loss = checkpoint['best_summary_loss']
    self.epoch = checkpoint['epoch'] + 1
      
  def log(self, message):
    if self.config.verbose:
      print(message)
    with open(self.log_path, 'a+') as logger:
      logger.write(f'{message}\n')
   
   
   
# effdet_runner

def effdet_runner():
  net.to(device)

  train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size = effdet_config.batch_size,
      sampler = RandomSampler(train_dataset),
      pin_memory = False,
      drop_last = True,
      num_workers = effdet_config.num_workers,
      collate_fn = collate_fn)
  
  valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset, 
      batch_size = effdet_config.batch_size,
      num_workers = effdet_config.num_workers,
      shuffle = False,
      sampler = SequentialSampler(valid_dataset),
      pin_memory = False,
      collate_fn = collate_fn)

  fitter = effdet_fitter(model = net, config = effdet_config)
  fitter.fit(train_dataloader, valid_dataloader)
