# this is to test the pgan derived from train
path_to_db = "../../../data/" 

import os
import sys
# path to the code
os.chdir("/media/AURA/Documents/ScriptsPrograms/ml_art/bin/gan_zoo_pytorch/pytorch_GAN_zoo/")
sys.path.append("/media/AURA/Documents/ScriptsPrograms/ml_art/bin/gan_zoo_pytorch/pytorch_GAN_zoo/")
import importlib
#import argparse
import json

from models.utils.utils import getVal, getLastCheckPoint, loadmodule
from models.utils.config import getConfigOverrideFromParser, updateParserWithConfig
import numpy as np

import albumentations as A
import cv2
from models.utils.image_transform import pil_loader, ToTensorV2

def getTrainer(name):
    
    match = {"PGAN": ("progressive_gan_trainer", "ProgressiveGANTrainer"),
             "StyleGAN":("styleGAN_trainer", "StyleGANTrainer"),
             "DCGAN": ("DCGAN_trainer", "DCGANTrainer")}
    
    if name not in match:
        raise AttributeError("Invalid module name")
    
    return loadmodule("models.trainer." + match[name][0],match[name][1],prefix='')

model_name = 'PGAN'
trainerModule = getTrainer(model_name)
vis_module = importlib.import_module("visualization.np_visualizer")
# max number of iterations per scale
my_maxIterAtScale = [1200,1600,1600,1600]+[2000]*5
 # If _C.alphaJumpMode == "linear", then the following fields should be completed
# Number of jumps per scale
#_C.alphaNJumps = [0, 600, 600, 600, 600, 600, 600, 600, 600]
my_alphaNJumps = [0]+[50]*8
# Number of iterations between two jumps
#_C.alphaSizeJumps = [0, 32, 32, 32, 32, 32, 32, 32, 32, 32]
my_alphaSizeJumps = [0, 32, 32, 32, 32, 32, 32, 32, 32, 32]
my_bs =16
kwargs = {'model_name': 'PGAN', 'no_vis': False, 'np_vis': True, 'restart': False, 'name': 'test4', 'dir': '../output_networks', 'configPath': 'config_test.json', 'saveIter': 16000, 'evalIter': 100, 'Scale_iter': None, 'partition_value': None, 'maxIterAtScale': my_maxIterAtScale, 'alphaJumpMode': 'linear', 'iterAlphaJump': None, 'alphaJumpVals': None, 'alphaNJumps': my_alphaNJumps, 'alphaSizeJumps': my_alphaSizeJumps, 'depthScales': None, 'miniBatchSize': my_bs, 'dimLatentVector': None, 'initBiasToZero': None, 'perChannelNormalization': None, 'lossMode': None, 'lambdaGP': None, 'leakyness': None, 'epsilonD': None, 'miniBatchStdDev': None, 'baseLearningRate': None, 'dimOutput': None, 'weightConditionG': None, 'weightConditionD': None, 'GDPP': None, 'overrides': False}
# the nEpochs isn't clear what it is
# see ./models/trainer/standard_configurations/dcgan_config.py:50:_C.nEpoch = 10
trainingConfig = {'config': {}}
#arguments for GANTrainer = trainerModule 'visualisation':need to import vis_module = importlib.import_module("visualization.np_visualizer"),'lossIterEvaluation':100,'checkPointDir':../output_networks/test1,'saveIter':16000,'modelLabel':test1,'partitionValue':None

# Build the output durectory if necessary
if not kwargs.get('dir','.'): 
    os.mkdir(kwargs.get('dir','.'))

# Checkpoint data
modelLabel = kwargs["name"]
restart = kwargs["restart"]
checkPointDir = os.path.join(kwargs["dir"], modelLabel)
checkPointData = getLastCheckPoint(checkPointDir, modelLabel)

if not os.path.isdir(checkPointDir):
    os.mkdir(checkPointDir)

with open(kwargs["configPath"], 'rb') as file:
    trainingConfig = json.load(file)

trainingConfig['pathDB'] = '../../../data/'
trainingConfig['imagefolderDataset'] = True

# Model configuration
configOverride = getConfigOverrideFromParser(kwargs, trainerModule._defaultConfig)
modelConfig = trainingConfig.get("config", {})
for item, val in configOverride.items():
    modelConfig[item] = val

trainingConfig["config"] = modelConfig
# Path to the image dataset
pathDB = trainingConfig["pathDB"]
trainingConfig.pop("pathDB", None)
partitionValue = getVal(kwargs, "partition_value", trainingConfig.get("partitionValue", None))

### Omnibus class that does all the transformations, see if it can be passed to DataSet
class AlbumentationsTransformations(object):
    def __init__(self, size):
        if isinstance(size,int):
            size = (size,size)
        self.A_Compose = A.Compose([A.RandomGridShuffle(always_apply=False, p=0.05, grid=(2, 2)),
                       A.transforms.GridDistortion(num_steps=5, distort_limit=0.4, p=0.5),
                       A.HorizontalFlip(p=0.5),
                       A.RandomResizedCrop(always_apply=True, p=1.0, height=size[0], width=size[0],
                                         scale=(0.75, 1),
                                         ratio=(0.98, 1.02), interpolation=3),#ONLY 3 SEEMS TO WORK
                       A.transforms.ColorJitter(brightness=0.05, contrast=0.07, saturation=0.04, hue=0.07, always_apply=False, p=0.6),#HSV random
                       A.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ToTensorV2()
                       ])
    
    def __call__(self, PIL_image):
        transformed_image_dict = self.A_Compose(image = np.array(PIL_image))
        return transformed_image_dict['image']
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

GANTrainer = trainerModule(pathDB,
                           #useGPU=True,
                           useGPU=False,
                           visualisation=vis_module,
                           lossIterEvaluation=kwargs["evalIter"],
                           checkPointDir=checkPointDir,
                           saveIter= kwargs["saveIter"],
                           modelLabel=modelLabel,
                           partitionValue=partitionValue,
                           albumentations_transformations = AlbumentationsTransformations,
                           **trainingConfig)
# Maximum number of iteration at each scale
print('maxIterAtScale');print(GANTrainer.modelConfig.maxIterAtScale) # 9 values
# [48000, 96000, 96000, 96000, 96000, 96000, 96000, 96000, 200000]
# Blending mode# 2 possible values are possible:
# - custom: iterations at which alpha should be updated and new value after the update are fully described by the user
# - linear: The user just inputs the number of updates of alpha, and the number of iterations between two updates for each scale
print('alphaJumpMode:');print(GANTrainer.modelConfig.alphaJumpMode) # linear
print('iterAlphaJump:');print([len(t) for t in GANTrainer.modelConfig.iterAlphaJump])
# lens: [1, 601, 601, 601, 601, 601, 601, 601, 601]
# values:[[0], [0, 32,..., 19168, 19200],...,[0, 32,..., 19168, 19200]]
print('alphaNJumps:');print(GANTrainer.modelConfig.alphaNJumps)

GANTrainer.train()

# errors torch.nn.modules.module.ModuleAttributeError: 'GNet' object has no attribute 'module'
# I changed:
# > File "/media/AURA/Documents/ScriptsPrograms/ml_art/bin/gan_zoo_pytorch/pytorch_GAN_zoo/models/progressive_gan.py", line 134, in updateAlpha
#    self.avgG.module.setNewAlpha(newAlpha) # Old
#    self.avgG.setNewAlpha(newAlpha) # New
