# coding: utf-8

from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from base.constants import *
from base.helpers import *
#from src.base.vizwiz_eval_cap.eval import VizWizEvalCap
#from dataset import DemoDataset   ## This is a local import from dataset.pyA
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from PIL import Image
import matplotlib.pyplot as plt
import os
import json

dataDir='/projectnb/ds598/projects/VizWiz-VQA-ds598/data'
split = 'train'
annFile='%s/annotations/%s.json'%(dataDir, split)
imgDir = '%s/train/' %dataDir

# initialize VQA api for QA annotations
vqa=VQA(annFile)

# load and display QA annotations for given answer types
"""
ansTypes can be one of the following
yes/no
number
other
unanswerable
"""
anns = vqa.getAnns(ansTypes='yes/no');   
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])
imgFilename = randomAnn['image']
if os.path.isfile(imgDir + imgFilename):
	I = io.imread(imgDir + imgFilename)
	#plt.imshow(I)
	#plt.axis('off')
	#plt.show()

# load and display QA annotations for given images
imgs = vqa.getImgs()
anns = vqa.getAnns(imgs=imgs)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])  
imgFilename = randomAnn['image']
if os.path.isfile(imgDir + imgFilename):
	I = io.imread(imgDir + imgFilename)
	#plt.imshow(I)
	#plt.axis('off')
	#plt.show()
