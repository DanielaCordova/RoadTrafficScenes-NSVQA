# Copyright © 2019 by Spectrico
# Licensed under the MIT License
# Based on the tutorial by Adrian Rosebrock: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
# Usage: $ python car_color_classifier_yolo3.py --image cars.jpg

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import color_classification
import model_classification

start = time.time()
dataDir = 'E/TFM/COCO/images/train2014/COCO_train2014_000000150559.jpg'
versionType = 'v2_'  # this should be '' when using VQA v2.0 dataset
taskType = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType = 'train2014'
annFile = '%s/Annotations/%s%s_%s_annotations_filtered.json' % (dataDir, versionType, dataType, dataSubType)
quesFile = '%s/Questions/%s%s_%s_%s_questions_filtered.json' % (dataDir, versionType, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)

weights="F:/Github/YOLOv3_PyTorchF/weights"
config= "F:/Github/COCO NSVQA/utils/config"
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",  default="E:/TFM/COCO/images/train2014/a2.jpg",
                help="path to input image")
ap.add_argument("-y", "--yolo", default='E:/TFM/COCO/data',
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

color_classifier = color_classification.Classifier()
model_classifier = model_classification.Classifier()

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 195, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([weights, "yolov3.weights"])
configPath = os.path.sep.join([config, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
outputs = net.forward(output_layers)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in outputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                        args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        if classIDs[i] == 2:
            color_result = color_classifier.predict(image[max(y, 0):y + h, max(x, 0):x + w])
            model_result = model_classifier.predict(image[max(y, 0):y + h, max(x, 0):x + w])

            color_text = "{}: {:.4f}".format(color_result[0]['color'], float(color_result[0]['prob']))
            cv2.putText(image, color_text, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            make_text = "{}: {:.4f}".format(model_result[0]['make'], float(model_result[0]['prob']))
            cv2.putText(image, make_text, (x + 2, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            model_text = model_result[0]['model']
            cv2.putText(image, model_text, (x + 2, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            print("[OUTPUT] {}, {}, {}".format(make_text, model_text, color_text))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

# output image
cv2.imwrite("E:/TFM/outputClassifier/output3.jpg", image)

# show timing information on MobileNet classifier
end = time.time()
print("[INFO] classifier took {:.6f} seconds".format(end - start))
