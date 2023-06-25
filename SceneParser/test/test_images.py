# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random

import matplotlib
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from common.utils import non_max_suppression, bbox_iou
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

coco_labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression, bbox_iou

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]


def test(config):
    with open('E:/TFM/Dataset1Car/valid/_annotations.coco.json') as json_file:
        dataset = json.load(json_file)

    categories = dataset.get("categories")
    dicCat = {}
    dicCatBac = {}
    for c in categories:
        dicCat[c.get("id")] = c.get("name")
        dicCatBac[c.get("name")] = c.get("id")

    dicCatBac["others"]=6
    annot = dataset.get("annotations")
    img_all = dataset.get("images")

    dicImg ={}
    dicAnn={}
    for i in img_all:
        dicImg[i["id"]]=i

    for an in annot:
        if dicAnn.__contains__(an["image_id"]):
            dicAnn[an["image_id"]].append(an)
        else:
            dicAnn[an["image_id"]]=[]
            dicAnn[an["image_id"]].append(an)

    dicFileNameId={}

    for i in img_all:
        dicFileNameId[i["file_name"]]=i["id"]

    is_training = False
    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # prepare images path
    images_name = os.listdir(config["images_path"])
    images_path = [os.path.join(config["images_path"], name) for name in images_name if "jpg" in name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["images_path"]))

    # Start inference
    batch_size = config["batch_size"]
    n_gt = 0
    correct = 0
    pred=[]
    real=[]
    labelsF=set()
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        images_origin = []
        images_names = []

        for path in images_path[step: (step-1) + batch_size]:
           
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            images_names.append(path.replace("E:/TFM/Dataset1Car/valid/", ""))

            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin.append(image)  # keep for save result
            image = cv2.resize(image, (config["img_w"], config["img_h"]),
                               interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        # inference
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, config["yolo"]["classes"],
                                                   conf_thres=config["confidence_threshold"],
                                                   nms_thres=0.45)

        # write result images. Draw bounding boxes and labels of detections
        classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]
        if not os.path.isdir("./output/"):
            os.makedirs("./output/")
        correctb=0
        totalB = 0
        cont=0
        for idx, detections in enumerate(batch_detections):
            cont+=1
            if detections is not None:
                unique_labels = detections[:, -1].detach().cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                found = False
                correctD=0
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Rescale coordinates to original dimensions
                    ori_h, ori_w = images_origin[idx].shape[:2]
                    pre_h, pre_w = config["img_h"], config["img_w"]
                    y2 = ((y2 - y1) / pre_h) * ori_h
                    x2 = ((x2 - x1) / pre_w) * ori_w
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w
                    
                    box_pred = torch.cat([coord.unsqueeze(0) for coord in [x1, y1, x2, y2]]).view(1, -1)

                    name = images_names[idx]
                    id = dicFileNameId[name]

                    found=False
                    for a in dicAnn[id]:
                        if dicCat[a["category_id"]]==classes[int(cls_pred)] and not found:
                            tx1, ty1, tx2, ty2 = a[ "bbox"]

                            ori_h, ori_w = images_origin[idx].shape[:2]
                            pre_h, pre_w = config["img_h"], config["img_w"]
                            ty2= torch.tensor(((ty2 - ty1) / pre_h) * ori_h)
                            tx2 = torch.tensor(((tx2 - tx1) / pre_w) * ori_w)
                            ty1 = torch.tensor((ty1 / pre_h) * ori_h)
                            tx1 = torch.tensor((tx1 / pre_w) * ori_w)

                            box_gt = torch.cat([coord.unsqueeze(0) for coord in [tx1, ty1, tx2, ty2]]).view(1, -1).cuda()
                            iou = bbox_iou(box_pred, box_gt)
                            if iou >= config["iou_thres"]:
                                correct += 1
                                correctb+=1
                                correctD+=1
                                found=True
                        else:
                            print("Pred:" + dicCat[a["category_id"]] +" coco:" +classes[int(cls_pred)])

                    pred.append(classes[int(cls_pred)])
                    real.append(dicCat[a["category_id"]])
                    labelsF.add(classes[int(cls_pred)])


            print('Img [%d/%d] mAP: %.5f' % (cont-1, len(images_path), float(correctD / len(dicAnn[id]))))

            totalB +=len(dicAnn[id])

        print('Batch [%d/%d] mAP: %.5f' % (step, len(images_path), float(correctb / totalB)))

        n_gt+= totalB

    print('Mean Average Precision: %.5f' % float(correct / n_gt))

    import pandas as pd


    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(real, pred, labels=list(labelsF))
    # Print the confusion matrix using Matplotlib
    #
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(font_scale=0.5)
    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, fmt='g', ax=ax)  # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labelsF)
    ax.yaxis.set_ticklabels(labelsF)


    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    logging.info("Save all results to ./output/")


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python test_images.py params.py")
        sys.exit()
    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    test(config)


if __name__ == "__main__":
    main()
