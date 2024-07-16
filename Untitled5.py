#!/usr/bin/env python
# coding: utf-8

# # Loading Libraries

# In[12]:

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
parser.add_argument('--text_threshold', default=0.01, type=float, help='text confidence threshold')  # 0.7
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')  # 0.4
parser.add_argument('--link_threshold', default=0.05, type=float, help='link confidence threshold')  # 0.2  0.05
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1920, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.0, type=float, help='image magnification ratio')  # 1.5
parser.add_argument('--poly', default=True, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='Data2', type=str, help='folder path to input images')
parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                    help='pretrained refiner model')

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
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

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

    if args.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


# In[27]:


def loadCraft():
    # load net
    global net
    net = CRAFT()  # initialize

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
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
                                             args.cuda, args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))


# In[28]:


loadCraft()


def polygon_to_rectangle(polygon):
    # Extract x and y coordinates
    x_coords = polygon[::2]
    y_coords = polygon[1::2]

    # Find minimum and maximum x and y coordinates
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # Construct the rectangle with 8 coordinates
    rectangle = [
        min_x, min_y,  # Top-left corner
        max_x, min_y,  # Top-right corner
        max_x, max_y,  # Bottom-right corner
        min_x, max_y  # Bottom-left corner
    ]

    return rectangle


def read_bounding_boxes_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines_cleaned = [line.strip() for line in lines if line.strip()]
        boxes = []
        for line in lines_cleaned:
            # Split the line by commas and remove any leading/trailing whitespace
            values = [value.strip() for value in line.strip().split(',') if value.strip()]

            # Convert non-empty values to integers
            box = list(map(int, values))
            if (len(box) > 8):
                box = polygon_to_rectangle(box)

            if box:  # Only append non-empty boxes
                boxes.append(box)

    return boxes


# #Loading TROCR

# In[29]:


from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from PIL import Image, ImageFilter

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")


# #Preprocess Image

# In[30]:
def read_bounding_boxes_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines_cleaned = [line.strip() for line in lines if line.strip()]
        boxes = []
        for line in lines_cleaned:
            # Split the line by commas and remove any leading/trailing whitespace
            values = [value.strip() for value in line.strip().split(',') if value.strip()]

            # Convert non-empty values to integers
            box = list(map(int, values))
            if (len(box) > 8):
                box = polygon_to_rectangle(box)
            if box:  # Only append non-empty boxes
                boxes.append(box)

    return boxes


def increase_boxes_size(bounding_boxes, increase_value):
    increased_boxes = []
    for box in bounding_boxes:
        # Extract coordinates
        x1, y1, x2, y2, x3, y3, x4, y4 = box

        # Calculate original height
        height = max(y3, y4) - min(y1, y2)

        # Increase height by the specified value
        increased_height = height + increase_value

        # Adjust the y-coordinates of the bottom vertices (y3, y4) for height increase
        y3 += increase_value
        y4 += increase_value

        # Append the modified bounding box to the list
        increased_boxes.append((x1, y1, x2, y2, x3, y3, x4, y4))
    return increased_boxes


def increase_bbox_width(boxes, image_width):
    increased_boxes = []
    for box in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        # Increase width by adding the increase value to x2 and x3 coordinates
        increased_box = (0, y1, image_width, y2, 0, y3, image_width, y4)
        increased_boxes.append(increased_box)
    return increased_boxes


# In[31]:


def crop_images(image_path, increased_boxes):
    image = Image.open(image_path)
    img_cropped = []
    for i in range(len(increased_boxes)):  # len(result2)-1
        x1, y1, x2, y2, x3, y3, x4, y4 = increased_boxes[i]
        left = min(x1, x2, x3, x4)
        upper = min(y1, y2, y3, y4)
        right = max(x1, x2, x3, x4)
        lower = max(y1, y2, y3, y4)
        cropped_width = right - left
        cropped_height = lower - upper
        if cropped_width >= 20 and cropped_height >= 20:
            img_cropped.append(image.crop((left, upper, right, lower)))

    return img_cropped


def save_cropped_images(img_cropped):
    for i in range(0, len(img_cropped)):
        img_cropped[i].save(f'Cropped/Test_Image{i}.jpg')


def levenshtein_distance(s1, s2):
    """
    Compute Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def are_similar(s1, s2, threshold):
    """
    Check if two strings are similar based on Levenshtein distance.
    """
    distance = levenshtein_distance(s1, s2)
    similarity = 1 - distance / max(len(s1), len(s2))
    return similarity > threshold


# In[32]:


import re


def remove_non_alphabet_characters(input_string):
    result_string = re.sub(r'[^a-zA-Z\s]', '', input_string)
    return result_string


def TROCR(img_cropped, resultant_string=""):
    strings = []
    for i in range(0, len(img_cropped)):
        image = Image.open(f'Cropped/Test_Image{i}.jpg')

        # calling the processor is equivalent to calling the feature extractor
        pixel_values = processor(image, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_text = remove_non_alphabet_characters(generated_text)
        generated_text = generated_text.strip()

        if (i > 0 and are_similar(strings[len(strings) - 1], generated_text, 0.5)):
            print("check")

        else:
            strings.append(generated_text)
            resultant_string += generated_text
            print(f"{i}:{generated_text}")

        resultant_string += " "
    return resultant_string


def remove_files_in_directory(directory):
    # Get list of files in the directory
    files = os.listdir(directory)

    # Iterate through each file and remove it
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}")


# # FAST API

# In[33]:


import threading
import uvicorn

from fastapi import FastAPI
from fastapi import File, UploadFile, Form
import io

app = FastAPI()


@app.get('/')
async def read_root():
    return "Hello, FastAPI!"


@app.post('/upload/')
async def image_upload(file: UploadFile = File(...)):
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
        remove_files_in_directory('Data2')
        await image_upload(file)

        runCraft()

        try:
            image_path = file.filename
            if image_path.endswith(".jpg"):
                image_path = image_path[:-4]
            word_boxes = read_bounding_boxes_from_file(f'result/res_{image_path}.txt')
            print("here1")


        except Exception:
            return {"message": f'result/res_{image_path}.txt'}

        try:

            file_path = os.path.join('result', f"res_{file.filename}")

            global image

            print(image_path)
            image = Image.open(f"result/res_{image_path}.jpg")

            increased_boxes1 = increase_bbox_width(word_boxes, image.width)

        except Exception:
            return {"message": f"Error Increasing Box Width {image.width}"}

        try:
            increased_boxes = increase_boxes_size(increased_boxes1, 10)

        except Exception:
            return {"message": "Error Increasing Box Size"}

        try:
            file_path = os.path.join('Data2', file.filename)  ## change folder
            print("here7")
            img_cropped = crop_images(file_path, increased_boxes)
            save_cropped_images(img_cropped)
            print("here8")
        except Exception:
            return {"message": "Could Not Crop Image"}

        resultant_string = TROCR(img_cropped)

        return resultant_string



    except Exception:
        return {"message": "There was an error in converting image to handwritten Text"}

# In[ ]:


# In[ ]:




