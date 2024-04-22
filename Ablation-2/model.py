# %%
import warnings
import os
import os
import torch
from torchvision.datasets import CIFAR100
import math
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from PIL import Image
from torchvision import transforms
from PIL import ImageFile
import json
from tqdm.auto import tqdm
import random
from pytorch_lightning import seed_everything
ImageFile.LOAD_TRUNCATED_IMAGES = True


# %%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

# %%
data = pd.read_csv('/MAMI_train_set.csv')  #Drive link where your data is saved
data_test = pd.read_csv('/MAMI_test_set.csv')
scene_train = pd.read_csv('/MAMI_scene_graph_train.csv')
scene_test=pd.read_csv('/MAMI_test_scene_graph.csv')
llm_data = pd.read_excel('/mistral_training.xlsx')

# %%
# dataset_folder_path = '/media/kuchbhi/D-Drive1/mami/MAMI_data/training_images/'
dataset_folder_path = '/MAMI_2022_images/training_images'


# %%

def get_data(data):
  #data = pd.read_csv(dataset_path)
  text = list(data['Text_Transcription'])
  img_path = list(data['file_name'])
  name = list(data['file_name'])
  # label = list(data['Level1'])
  # label = list(data['overall_sentiment'])
  label = list(data['misogynous'])

  text_features,emotion_features,rationale_features, context_features,image_features,l,Name,v= [],[],[],[],[],[],[],[]

  for txt,img,L,n in tqdm(zip(text,img_path,label,name)):
    img2=copy.deepcopy(img)
    try:
      img = Image.open(dataset_folder_path+img)
    except Exception as e:
      print(e)
      continue
    try:
      row=rag_data.loc[rag_data['file_name']==img2].iloc[0]
    except Exception as e:
      print(e)
      continue
    emotion, rationale, context2 = row['emotion'], row['rationale'], row['context']
    context = context2.replace('?',' ')
    if(len(emotion)<20 or len(rationale)<20 or len(context)<20):continue
    img = torch.stack([compose(img).to(device)])
    l.append(L)
    Name.append(n)
    with torch.no_grad():
      temp_txt=model.forward(txt, tokenizer).detach().cpu().numpy()
      temp_emotion = model.forward(emotion, tokenizer).detach().cpu().numpy()
      temp_rationale = model.forward(rationale, tokenizer).detach().cpu().numpy()
      temp_context = model.forward(context, tokenizer).detach().cpu().numpy()
      temp_img = clip_model.encode_image(img).detach().cpu().numpy()
      image_features.append(temp_img)
      text_features.append(temp_txt)
      emotion_features.append(temp_emotion)
      context_features.append(temp_context)
      rationale_features.append(temp_rationale)
      del temp_txt
      del temp_img
      del temp_emotion
      del temp_rationale
      del temp_context
      torch.cuda.empty_cache()
    del img
    torch.cuda.empty_cache()
  return text_features,image_features,emotion_features,rationale_features, context_features,l,Name


class MAMIDataset(Dataset):

  def __init__(self,data):
    self.t_f,self.i_f,self.e_f,self.r_f,self.c_f,self.label,self.name = get_data(data)
    self.t_f = np.squeeze(np.asarray(self.t_f),axis=1)
    self.i_f = np.squeeze(np.asarray(self.i_f),axis=1)
    self.e_f = np.squeeze(np.asarray(self.e_f),axis=1)
    self.r_f = np.squeeze(np.asarray(self.r_f),axis=1)
    self.c_f = np.squeeze(np.asarray(self.c_f),axis=1)
    # self.t_f = np.squeeze(np.asarray(self.t_f))
    # self.i_f = np.squeeze(np.asarray(self.i_f))
    # self.e_f = np.squeeze(np.asarray(self.e_f))
    # self.r_f = np.squeeze(np.asarray(self.r_f))
    # self.c_f = np.squeeze(np.asarray(self.c_f))
  def __len__(self):
    return len(self.label)

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    #print(idx)
    name=self.name[idx]
    label = self.label[idx]
    T = self.t_f[idx,:]
    I = self.i_f[idx,:]
    E = self.e_f[idx,:]
    R = self.r_f[idx,:]
    C = self.c_f[idx,:]
    sample = {'label':label,'processed_txt':T,'processed_img':I,'processed_emotion':E,'processed_rationale':R,'processed_context':C,'name':name}
    return sample

# %%
training_dataset = torch.load('/mistral_training.pt')
testing_dataset = torch.load('/mistral_testing.pt')

# %%
torch.manual_seed(42)
total = len(training_dataset)
vpsz = total//4
t_p,v_p = torch.utils.data.random_split(training_dataset,[total-vpsz,vpsz])
te_p = testing_dataset

# %%
print(len(t_p))
print(len(v_p))
print(len(te_p))
print(len(training_dataset))

# %%
pred=[]
df_pred = pd.DataFrame(columns=['Name', 'Text_Transcription', 'Predicted Label', 'True Label', 'Class'])
ff=[]

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
class MFB(nn.Module):
    def __init__(self,img_feat_size, ques_feat_size, is_first, MFB_K, MFB_O, DROPOUT_R):
        super(MFB, self).__init__()
        #self.__C = __C
        self.MFB_K = MFB_K
        self.MFB_O = MFB_O
        self.DROPOUT_R = DROPOUT_R

        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)

        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride = MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        exp_out = img_feat * ques_feat             # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.MFB_O)      # (N, C, O)
        return z

# %%
class Classifier(pl.LightningModule):
  def __init__(self):
      super().__init__()
      self.MFB = MFB(512, 3072, True, 256, 768, 0.1)
      self.MFB2 = MFB(768, 768, True, 256, 768, 0.1)
      self.fin_y_shape = nn.Linear(768, 512)
      self.fin_old = nn.Linear(64, 2)
      self.fin = nn.Linear(16 * 768, 64)
      self.fin_2 = nn.Linear(128,768)
      self.fin_3=nn.Linear(768,2)
      self.validation_step_outputs = []
      self.test_step_outputs = []

  def forward(self, x, y, e, r, c):
      con = torch.cat((x,e,r,c),dim=1)
      fusion = self.MFB(torch.unsqueeze(y, axis=1), torch.unsqueeze(con, axis=1))
      fusion_tmp = torch.squeeze(fusion, dim=1)
      output = torch.log_softmax(fusion_tmp, dim=1)
      return output,output,output,output

  def cross_entropy_loss(self, logits, labels):return F.nll_loss(logits, labels)
  def contrastive_loss(self, y1, y2, d=0, m=2.0):
  # d = 0 means y1 and y2 are supposed to be same
  # d = 1 means y1 and y2 are supposed to be different

    euc_dist = F.pairwise_distance(y1, y2)

    if d == 0:
      return torch.mean(torch.pow(euc_dist, 2))  # distance squared
    else:  # d == 1
      delta = m - euc_dist  # sort of reverse distance
      delta = torch.clamp(delta, min=0.0, max=None)
      return torch.mean(torch.pow(delta, 2))  # mean over all rows
  def training_step(self, train_batch, batch_idx):
      lab,txt,img,emotion,rationale,context,name= train_batch
      lab = train_batch[lab]
      #print(lab)
      txt = train_batch[txt]
      img = train_batch[img]
      emotion = train_batch[emotion]
      rationale = train_batch[rationale]
      context=train_batch[context]
      logits, emf, timr, ccmi= self.forward(txt,img,emotion,rationale,context)
      loss1 = self.cross_entropy_loss(logits, lab)
      #loss2 = self.contrastive_loss(emf,timr)
      loss3 = self.contrastive_loss(emf,ccmi)
      loss = loss1
      self.log('train_loss', loss)
      return loss


     
  def validation_step(self, val_batch, batch_idx):
      lab,txt,img,emotion,rationale,context,name= val_batch
      lab = val_batch[lab]
      #print(lab)
      txt = val_batch[txt]
      img = val_batch[img]
      emotion=val_batch[emotion]
      rationale=val_batch[rationale]
      context=val_batch[context]
      logits, emf, timr, ccmi= self.forward(txt,img,emotion,rationale,context)
      logits=logits.float()
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss1 = self.cross_entropy_loss(logits, lab)
      #loss2 = self.contrastive_loss(emf,timr)
      #loss3 = self.contrastive_loss(emf,ccmi)
      loss =loss1
      lab = lab.detach().cpu().numpy()

      # logits=logits.float()
      # tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      # loss = self.cross_entropy_loss(logits, lab)
      # lab = lab.detach().cpu().numpy()
      self.log('val_acc', accuracy_score(lab,tmp))
      self.log('val_roc_auc',roc_auc_score(lab,tmp))
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': accuracy_score(lab,tmp)}
      self.validation_step_outputs.append({'progress_bar': tqdm_dict,'val_f1': f1_score(lab,tmp,average='macro')})

      return {
                'progress_bar': tqdm_dict,
      'val_f1': f1_score(lab,tmp,average='macro')
      }

  def on_validation_epoch_end(self):
    outs = []
    outs14=[]
    for out in self.validation_step_outputs:
       outs.append(out['progress_bar']['val_acc'])
       outs14.append(out['val_f1'])
    self.log('val_acc_all_offn', sum(outs)/len(outs))
    self.log('val_f1', sum(outs14)/len(outs14))
    print(f'***val_acc_all_offn at epoch end {sum(outs)/len(outs)}****')
    print(f'***val_f1  at epoch end {sum(outs14)/len(outs14)}****')
    self.validation_step_outputs.clear()

  def test_step(self, batch, batch_idx):
    #   lab,txt,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12, e13,e14, e15,e16,img,name= batch
      global df_pred
      global pred
      global ff
      lab,txt,img,emotion,rationale,context,name= batch
      lab = batch[lab]
      #print(lab)
      name2=batch[name]
      txt = batch[txt]
      img = batch[img]
      emotion = batch[emotion]
      rationale=batch[rationale]
      context=batch[context]
      logits, emf, timr, ccmi= self.forward(txt,img,emotion,rationale,context)
      logits = logits.float()
      tmp = np.argmax(logits.detach().cpu().numpy(force=True),axis=-1)
      loss1 = self.cross_entropy_loss(logits, lab)
      #loss2 = self.contrastive_loss(emf,timr)
      loss3 = self.contrastive_loss(emf,ccmi)
      loss = loss1
      lab = lab.detach().cpu().numpy()
      for i in range(len(name2)):
        file_named = name2[i]
        ff.append(file_named)
        predicted_label = tmp[i]
        row = data_test.loc[data_test['file_name'] == file_named].iloc[0]
        true_label = row['misogynous']
        tt1 = row['Text_Transcription']
        #print(true_label)
        if true_label==1 and predicted_label==1:
          res="True Positive"
        if true_label==0 and predicted_label==0:
          res="True Negative"
        if true_label==1 and predicted_label==0:
          res="False Negative"
        if true_label==0 and predicted_label==1:
          res="False Positive"
        new2={'Name' : file_named , 'Predicted Label' : predicted_label , 'True Label' : true_label , 'Class' : res, 'Text_Transcription':tt1}
        new=pd.DataFrame([new2])
        pred.append(file_named)
        df_pred = pd.concat([df_pred,new], ignore_index=True)
      self.log('test_acc', accuracy_score(lab,tmp))
      # self.log('test_roc_auc',roc_auc_score(lab,tmp))
      self.log('test_loss', loss)
      tqdm_dict = {'test_acc': accuracy_score(lab,tmp)}
      self.test_step_outputs.append({'progress_bar': tqdm_dict,'test_acc': accuracy_score(lab,tmp), 'test_f1_score': f1_score(lab,tmp,average='macro')})
      return {
                'progress_bar': tqdm_dict,
                'test_acc': accuracy_score(lab,tmp),
                'test_f1_score': f1_score(lab,tmp,average='macro')
      }
  def on_test_epoch_end(self):
      # OPTIONAL
      outs = []
      outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
      [],[],[],[],[],[],[],[],[],[],[],[],[],[]
      for out in self.test_step_outputs:
        outs.append(out['test_acc'])
        outs2.append(out['test_f1_score'])
      self.log('test_acc', sum(outs)/len(outs))
      self.log('test_f1_score', sum(outs2)/len(outs2))
      self.test_step_outputs.clear()

  def configure_optimizers(self):
    # optimizer = torch.optim.Adam(self.parameters(), lr=3e-2)
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

    return optimizer


class HmDataModule(pl.LightningDataModule):

  def setup(self, stage):
    self.hm_train = t_p
    self.hm_val = v_p
    # self.hm_test = test
    self.hm_test = testing_dataset

  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=128, drop_last=True)

  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=128, drop_last=True)

  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128, drop_last=True)

data_module = HmDataModule()
checkpoint_callback = ModelCheckpoint(
     monitor='val_acc_all_offn',
     dirpath='checkpoints/',
     filename='epoch{epoch:02d}-val_f1_all_mis{val_acc_all_offn:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)
seed_everything(42, workers=True)
hm_model = Classifier()
trainer = pl.Trainer(deterministic=True,max_epochs=64,precision=16,callbacks=all_callbacks,accelerator="cuda")
trainer.fit(hm_model, data_module)

# %%
test_dataloader = DataLoader(dataset=testing_dataset, batch_size=8)
ckpt_path = '/LLM_COT_TEST/epoch50-val_f1_all_offn.ckpt' # put ckpt_path according to the path output in the previous cell
trainer.test(dataloaders=test_dataloader,ckpt_path=ckpt_path)

# %%
print(df_pred)

# %%
df_pred.to_excel('test_results.xlsx')


