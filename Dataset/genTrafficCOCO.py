# coding: utf-8

from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
import spacy
import nltk
from pattern.en import singularize, parse
from spacy import displacy
from spacy.matcher import DependencyMatcher
from matplotlib.colors import is_color_like
import pandas as pd

COCO_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

type1 = ["are in the picture?", "are there?", "are in this photo?", "can be seen?", "are visible?", "are pictured?",
         "do you see", "are shown?"]

place = ["bottom", "up", "above", "left", "right", "ground", "front", "behind", "next", "closest", "furthest", "under",
         "behind", "top", "down", "south", "north", "east", "west"]


def special_parse(word):
    if word == "people":
        return "person"
    else:
        return word


def get_question_features(question):
    ''' For a given question, a unicode string, returns the time series vector
	with each word (token) transformed into a 300 dimension representation
	calculated using Glove Vector '''
    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
        question_tensor[0, j, :] = tokens[j].vector
    return question_tensor


import json
from pathlib import Path
import cv2

def createCOCOtraffic(dataSubType):
    with open('E:/TFM/COCO/data/coco/Annotations/instances_train2014.json') as json_file:
        coco = json.load(json_file)
    imgIDs = set()

    for ann in coco.get('annotations'):
        if ann.get('category_id')==3 or ann.get('category_id')==4 or ann.get('category_id')==6 or ann.get('category_id')==8:
            imgIDs.add('COCO_' + dataSubType + '_' + str(ann.get('image_id')).zfill(12) + '.jpg')

    for im in imgIDs:
        img = cv2.imread('E:/TFM/COCO/images/train2014/'+im)
        cv2.imwrite('E:/TFM/COCO/images/train2014FILTERED/'+im, img)

def createVQAtraffic(dataSubType):
    with open('E:/TFM/COCO/data/coco/Annotations/instances_train2014.json') as json_file:
        coco = json.load(json_file)
    with open('E:/TFM/COCO/data/coco/Annotations/v2_mscoco_train2014_annotations.json') as json_file:
        anno = json.load(json_file)
    with open('E:/TFM/COCO/data/coco/Questions/v2_OpenEnded_mscoco_train2014_questions.json') as json_file:
        ques = json.load(json_file)

    imgIDs = set()

    for ann in coco.get('annotations'):
        if ann.get('category_id')==3 or ann.get('category_id')==4 or ann.get('category_id')==6 or ann.get('category_id')==8:
            imgIDs.add(ann.get('image_id'))

    ANN1={}
    Q1=[]
    for im in anno.get('annotations'):
        if im.get('image_id') in imgIDs:
            if ANN1.__contains__(im.get('image_id')):
                ANN1[im.get('image_id')][im.get('question_id')]=im
            else:
                ANN1[im.get('image_id')]={}
                ANN1[im.get('image_id')][im.get('question_id')] = im
    for im in ques.get('questions'):
        if im.get('image_id') in imgIDs:
            Q1.append(im)

    DataFrame = pd.DataFrame(ANN1)
    #DataFrame.to_csv('E:/TFM/COCO/data/coco/Annotations/v2_mscoco_train2014_annotations_DATASET19.csv', sep=',', index=None)

    DataFrame = pd.DataFrame(Q1)
    #DataFrame.to_csv('E:/TFM/COCO/data/coco/Questions/v2_OpenEnded_mscoco_train2014_questions_DATASET19.csv', sep=',', index=None)

    with open('E:/TFM/COCO/data/coco/Annotations/v2_mscoco_train2014_annotations_DATASET19.json', 'w') as f:
        json.dump(ANN1, f)
    with open('E:/TFM/COCO/data/coco/Annotations/v2_OpenEnded_mscoco_train2014_questions_DATASET19.json', 'r') as f:
        json.dump(Q1, f)




if __name__ == "__main__":
    #nlp = spacy.load('en_core_web_md')

    dataDir = 'F:/Github/YOLOv3_PyTorchF/data/coco'
    versionType = 'v2_'  # this should be '' when using VQA v2.0 dataset
    taskType = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
    dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
    dataSubType = 'train2014'
    annFile = '%s/Annotations/%s%s_%s_annotations_filtered.json' % (dataDir, versionType, dataType, dataSubType)
    quesFile = '%s/Questions/%s%s_%s_%s_questions_filtered.json' % (
        dataDir, versionType, taskType, dataType, dataSubType)
    imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)

    createVQAtraffic(dataSubType)

