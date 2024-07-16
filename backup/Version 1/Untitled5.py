#!/usr/bin/env python
# coding: utf-8

# # Loading Libraries

# In[12]:

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import sys

import time
import argparse


# In[13]:


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


# In[14]:


import craft_utils
import imgproc
import file_utils


# In[15]:


import json
import zipfile


# In[16]:


import cv2
from skimage import io
import numpy as np


# In[17]:


from PIL import Image


# In[18]:




# #Loading Craft Model
# 

# In[19]:


from craft import CRAFT


# In[20]:




from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


# In[21]:


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


# In[23]:


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold') #0.7
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.2, type=float, help='link confidence threshold') #0.2
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference') #Default is true to use gpu (gpu limit reached in colab)
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=2.0, type=float, help='image magnification ratio') # 1.5
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='Data2', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')


# In[24]:


args, unknown = parser.parse_known_args()



# In[25]:


result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


# In[26]:


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


# In[27]:

    


def loadCraft():
    # load net
    global net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    global refine_net
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True



def runCraft():
    
    image_list, _, _ = file_utils.get_files(args.test_folder)

    

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))


# In[28]:


loadCraft()


# #Loading TROCR

# In[29]:


from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from PIL import Image, ImageFilter
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")


# #Preprocess Image

# In[30]:


def calculate_threshold(boxes):
    vertical_gaps = []
    for i in range(1, len(boxes)):
        vertical_gap = boxes[i][1] - boxes[i-1][3]  
        vertical_gaps.append(vertical_gap)
    median_gap = np.median(vertical_gaps)
    threshold = median_gap * 5  # Adjust multiplier as needed
    print(f"threshold:{threshold}")
   
    return threshold

def group_boxes_by_line(boxes, text_threshold):
    lines = []
    sorted_boxes = sorted(boxes, key=lambda x: (x[1], x[0]))
    current_line = [sorted_boxes[0]]
    for box in sorted_boxes[1:]:
        if box[1] - current_line[-1][1] < text_threshold:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    lines.append(current_line)
    return lines
def read_bounding_boxes_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines_cleaned = [line.strip() for line in lines if line.strip()]
        boxes = []
        for line in lines:
            # Split the line by commas and remove any leading/trailing whitespace
            values = [value.strip() for value in line.strip().split(',') if value.strip()]
            # Convert non-empty values to integers
            box = list(map(int, values))
            if box:  # Only append non-empty boxes
                boxes.append(box)
    return boxes


# In[31]:


def crop_images(image_path,line_boxes):
    image = Image.open(image_path)
# Assuming `line_boxes` contains the line-level bounding boxes in the format (x_min, y_min, x_max, y_max)   
    cropped_images = []
    result2=[]
    print(image_path)
    print(len(line_boxes))
    for i, line in enumerate(line_boxes):
        x_min = 9999
        y_min= 9999
        x_max=-9999
        y_max=-9999

        for j, box in enumerate(line):
        # Assuming each box has 4 points
            x_min = min(box[0], box[2], box[4], box[6],x_min)
            y_min = min(box[1], box[3], box[5], box[7],y_min)
            x_max = max(box[0], box[2], box[4], box[6],x_max)
            y_max = max(box[1], box[3], box[5], box[7],y_max)
        result2.append((x_min,y_min,x_max,y_min,x_min,y_max,x_max,y_max))
    
    img_cropped=[]
    for i in range(len(result2)): #len(result2)-1
        x1, y1, x2, y2, x3, y3, x4, y4 = result2[i]
        left = min(x1, x4)
        upper = min(y1, y2)
        right = max(x2, x3)
        lower = max(y3, y4)
        img_cropped.append(image.crop((x1, y1, x4, y4)))
    for i in range (0,len(img_cropped)):
        img_cropped[i].save(f'Cropped/Test_Image{i}.jpg')
    return img_cropped







# In[32]:


import re

def remove_non_alphabet_characters(input_string):
    result_string = re.sub(r'[^a-zA-Z\s]', '', input_string)
    return result_string


def TROCR(img_cropped,resultant_string=""):
    for i in range (0,len(img_cropped)):
        image=Image.open(f'Cropped/Test_Image{i}.jpg')
        sharpened_img = image.filter(ImageFilter.SHARPEN)
        #processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        # calling the processor is equivalent to calling the feature extractor
        pixel_values = processor(sharpened_img, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_text=remove_non_alphabet_characters(generated_text)
        generated_text=generated_text.strip()
        resultant_string+=generated_text.lower()
        resultant_string+=" "
    return resultant_string
    


# # FAST API

# In[33]:



import threading
import uvicorn

from fastapi import FastAPI
from fastapi import File,UploadFile
import io

app = FastAPI()


@app.get('/')
async def read_root():
    return "Hello, FastAPI!"


@app.post('/upload/')
async def image_upload(file: UploadFile = File(...), name: str = None):
    try:
        os.makedirs('Data2', exist_ok=True)
        file_path = os.path.join('Data2', file.filename)
       
        with open(file_path, 'wb') as f:
            contents = await file.read()
            f.write(contents)
        
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}


@app.post('/imageToText/')
async def imageToText(file: UploadFile = File(...)):
    try:
        await image_upload(file)
        runCraft()
        try:
            image_path = file.filename
            if image_path.endswith(".jpg"):
                image_path = image_path[:-4]
            word_boxes = read_bounding_boxes_from_file(f'result/res_{image_path}.txt')
        except Exception:
            return {"message": f'result/res_{image_path}.txt'}
        try:
            dynamic_threshold = calculate_threshold(word_boxes)
            line_boxes = group_boxes_by_line(word_boxes, dynamic_threshold)
            
        except Exception:
            return {"message": "Processing Error"}
            
        try:
            file_path = os.path.join('Data2', file.filename) ## change folder
            img_cropped=crop_images(file_path,line_boxes)
        except Exception:
            return {"message": "Could Not Crop Image"}
        
        resultant_string=TROCR(img_cropped)
   
        return resultant_string
        
        
        
    except Exception:
        return {"message": "There was an error in converting image to handwritten Text"}
        
        











# In[ ]:




   


# In[ ]:




