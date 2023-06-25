# coding='utf-8'
import multiprocessing
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
import torchvision.transforms as transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import PIL.Image as Image

import Classificator.color_classification as color_classification
import Classificator.vehicle_classification as model_classification
import re
import torch
import torch.nn as nn
import cv2

currentPath = ""
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression, bbox_iou

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]


class Scene_Parser(nn.Module):
    '''Sequence to Sequence Model using Transformers'''

    def __init__(self, config):
        super(Scene_Parser, self).__init__()
        '''Initialize the model'''
        is_training = False
        self.config = config
        # Create Model, load
        self.net = ModelMain(config, is_training=is_training)
        self.net.train(is_training)

        # Set data parallel
        self.net = nn.DataParallel(self.net)
        self.net = self.net.cuda()

        # Restore pretrain model
        if self.config["pretrain_snapshot"]:
            logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
            self.state_dict = torch.load(config["pretrain_snapshot"])
            self.net.load_state_dict(self.state_dict)
        else:
            raise Exception("missing pretrain_snapshot!!!")

        # YOLO loss with 3 scales
        self.yolo_losses = []
        for i in range(3):
            self.yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                             config["yolo"]["classes"], (config["img_w"], config["img_h"])))

        self.color_classifier = color_classification.Classifier()
        self.model_classifier = model_classification.Classifier()

    def predict(self, img):

        # Load and initialize network

        # Start inference
        batch_size = self.config["batch_size"]
        QA = {}

        logging.info("processing: {}".format(img))
        image = cv2.imread(img, cv2.IMREAD_COLOR)
        if image is None:
            logging.error("read path error: {}. skip it.".format(img))
            exit()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_origin = []
        images_origin.append(image)  # keep for save result
        image = cv2.resize(image, (self.config["img_w"], self.config["img_h"]),
                           interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        images = []
        images.append(image)
        images_names_batch = []
        images_names_batch.append(img)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        # inference
        # print(images.shape[0])
        if images.shape[0] == 0:
            print("Error: " + path)
        else:
            with torch.no_grad():

                outputs = self.net(images)

                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_losses[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                       conf_thres=self.config["confidence_threshold"],

                                                       nms_thres=0.45)

                # write result images. Draw bounding boxes and labels of detections

                classes = open(self.config["classes_names_path"], "r").read().split("\n")[:-1]
                if not os.path.isdir("./output/"):
                    os.makedirs("./output/")
                count = 0
                image2 = cv2.imread(img)
                for idx, detections in enumerate(batch_detections):

                    plt.figure()
                    fig, ax = plt.subplots(1)
                    ax.imshow(images_origin[idx])
                    name = images_names_batch[idx].replace('E:/TFM/Dataset1Car/trainMod/', '')
                    QA3 = []
                    if detections is not None:
                        unique_labels = detections[:, -1].detach().cpu().unique()
                        n_cls_preds = len(unique_labels)
                        bbox_colors = random.sample(colors, n_cls_preds)
                        QAnew = []
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                            # Rescale coordinates to original dimensions
                            ori_h, ori_w = images_origin[idx].shape[:2]
                            pre_h, pre_w = self.config["img_h"], self.config["img_w"]
                            box_h = ((y2 - y1) / pre_h) * ori_h
                            box_w = ((x2 - x1) / pre_w) * ori_w
                            y1 = (y1 / pre_h) * ori_h
                            x1 = (x1 / pre_w) * ori_w

                            # Create a Rectangle patch
                            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                                     edgecolor=color,
                                                     facecolor='none')

                            # Add the bbox to the plot
                            ax.add_patch(bbox)
                            # Add label
                            plt.text(x1, y1, s=classes[int(cls_pred)], color='white',
                                     verticalalignment='top',
                                     bbox={'color': color, 'pad': 0})
                            x = int(x1.item())
                            y = int(y1.item())
                            w = int(x2.item())
                            h = int(y2.item())
                            numero = name.split("/")[-1]
                            numero = numero.replace('.jpg', '')
                            if int(cls_pred) == 2:
                                #
                                image = Image.open(img)
                                image= image.resize((self.config["img_w"], self.config["img_h"]))




                                x2 = min(x, w)
                                x1 = max(x, w)
                                y1 = min(y, h)
                                y2 = max(y, h)


                                color_result, c_conf = self.color_classifier.predict(
                                    image.crop((x2, y1, x1, y2)))
                                model_result, m_conf = self.model_classifier.predict(
                                    image.crop((x2, y1, x1, y2)))

                                color_text = "{}: {:.4f}".format(color_result, float(m_conf))
                                cv2.putText(image2, color_text, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                            1)

                                make_text = "{}: {:.4f}".format(model_result, float(m_conf))
                                cv2.putText(image2, make_text, (x + 2, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                model_text = make_text.split()[0]
                                cv2.putText(image2, model_text, (x + 2, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                            1)
                                print("[OUTPUT] {}, {}, {}".format(make_text, model_text, color_text))


                                model = model_result.split(" ")[-1]
                                company = model_result.replace(model,"")
                                company = company[:-1]
                                QAnew.append({'x1': x, 'y1': y, 'h': h, 'w': w,
                                              'class': classes[int(cls_pred)], 'color': color_result,
                                              'company': company,
                                              'model': model})
                                # plt.text(x1, y1, s=classes[int(cls_pred)], color='white',bbox={'color': color, 'pad': 0},
                                #          fontsize='xx-small', fontstretch='ultra-condensed',
                                #          multialignment='right', verticalalignment="bottom") #+ "\n"  + 'color:'+ color_text + "\n" +'type:' + make_text + "\n"+'model:' + model_result


                                color=(color[0]*100, color[1]*100, color[2]*100)
                                cv2.rectangle(image2, (x+h, y), (x, y+w), (0,0,0), 3)
#cv2.rectangle(image2, (x2, y1), (x1 + y2, y2 + x1), color, 3)



                            else:
                                image = Image.open(img)

                                x2 = min(x, w)
                                x1 = max(x, w)
                                y1 = min(y, h)
                                y2 = max(y, h)
                                color_result, c_conf = self.color_classifier.predict(
                                    image.crop((x2, y1, x1 + y2, y2 + x1)))

                                # image = cv2.imread(img)

                                color_text = "{}: {:.4f}".format(color_result, float(c_conf))
                                cv2.putText(image2, color_text, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                            1)


                                print("[OUTPUT] {}".format( color_text))
                                QAnew.append({'x1': x, 'y1': y, 'h': h, 'w': w,
                                              'class': classes[int(cls_pred)], 'color': color_result})
                                plt.text(x1, y1, s=classes[int(cls_pred)]+ "\n"'color:' + color_text, color='white',
                                         bbox={'color': color, 'pad': 0}, fontsize='xx-small', fontstretch='ultra-condensed',
                                         multialignment='right', verticalalignment="bottom")


                            #QAnew.append({'x1': x1.item(), 'y1': y1.item(), 'class': classes[int(cls_pred)]})

                        #QA3.append({'image_id': images_name[idx], 'x1': x1.item(), 'y1': y1.item(),'class': classes[int(cls_pred)]})

                        # if(int(numero)) in QA :
                        #     QA[int(numero)].append({ 'x1': x1.item(), 'y1': y1.item(), 'class': classes[int(cls_pred)]})
                        # else:
                        #     QA[int(numero)]=[]
                        #
                        #     QA[int(numero)].append({'x1': x1.item(), 'y1': y1.item(), 'class': classes[int(cls_pred)]})
                        plt.axis('off')
                        plt.savefig('output/{}'.format(name), bbox_inches='tight', pad_inches=0.0)

                        if "COCO" in numero:
                            numero=numero.split("_")[-1]
                        numero = numero.split("_")[0]
                        QA[str(numero)] = QAnew

                # Save generated image with detections
                cv2.imwrite("E:/TFM/outputYolo/{}_modelsSP_.jpg".format(0), image2)
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                plt.savefig('E:/TFM/outputYolo/{}_FGDV2'.format(0), bbox_inches='tight', pad_inches=0.0)
                plt.close()

                # print("Number of processors: ", multiprocessing.cpu_count())

        # with open(
        #         'F:/Github/COCO NSVQA/SceneParser/sceneParse1.json',
        #         'w') as f:
        #     json.dump(QA, f)

        logging.info("Save all results to ./output/")

        return QA.get(str(numero))

    def trainModel(self, img):

        # Load and initialize network

        # Start inference
        batch_size = self.config["batch_size"]
        output_array = []

        image = cv2.imread(img, cv2.IMREAD_COLOR)
        if image is None:
            logging.error("read path error: {}. skip it.".format(img))
            exit()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_origin = []
        images_origin.append(image)  # keep for save result
        image = cv2.resize(image, (self.config["img_w"], self.config["img_h"]),
                           interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        images = []
        images.append(image)
        images_names_batch = []
        images_names_batch.append(img)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        # inference
        # print(images.shape[0])
        if images.shape[0] == 0:
            print("Error: " + path)
        else:
            with torch.no_grad():

                outputs = self.net(images)

                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_losses[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                       conf_thres=self.config["confidence_threshold"],

                                                       nms_thres=0.45)

                # write result images. Draw bounding boxes and labels of detections

                classes = open(self.config["classes_names_path"], "r").read().split("\n")[:-1]
                if not os.path.isdir("./output/"):
                    os.makedirs("./output/")

                for idx, detections in enumerate(batch_detections):

                    plt.figure()
                    fig, ax = plt.subplots(1)
                    ax.imshow(images_origin[idx])
                    name = images_names_batch[idx].replace('E:/TFM/IDD_FGVD/train/images/', '')
                    QA3 = []
                    if detections is not None:
                        unique_labels = detections[:, -1].detach().cpu().unique()
                        n_cls_preds = len(unique_labels)
                        bbox_colors = random.sample(colors, n_cls_preds)
                        QAnew = []
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                            # Rescale coordinates to original dimensions
                            ori_h, ori_w = images_origin[idx].shape[:2]
                            pre_h, pre_w = self.config["img_h"], self.config["img_w"]
                            box_h = ((y2 - y1) / pre_h) * ori_h
                            box_w = ((x2 - x1) / pre_w) * ori_w
                            y1 = (y1 / pre_h) * ori_h
                            x1 = (x1 / pre_w) * ori_w

                            x = int(x1.item())
                            y = int(y1.item())
                            w = int(x2.item())
                            h = int(y2.item())
                            numero = name.split("/")[-1]
                            numero = numero.replace('.jpg', '')

                            #
                            image = Image.open(img)

                            x2 = min(x, w)
                            x1 = max(x, w)
                            y1 = min(y, h)
                            y2 = max(y, h)

                            outputs = self.model_classifier.model_ft(image.crop((x2, y1, x1 + y2, y2 + x1)))
                            output_array.append(outputs)

        return output_array


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
    Scene_Parser(config)


if __name__ == "__main__":
    main()
