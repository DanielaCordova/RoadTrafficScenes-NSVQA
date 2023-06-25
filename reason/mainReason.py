from programExecutor import *
import sys
import os
import json

sys.path.append('../')
from Questions.semanticParser2 import *
from SceneParser.scene_parser.scene_representation import *
import torch
from skimage.io import imshow
import cv2
import json
from pathlib import Path

import traceback
import logging
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import time
import os
import PIL.Image as Image
from IPython.display import display
import torchvision.transforms as T

newModels = set()
train_csv1 = 'E:/TFM/Dataset1Car/datasetQuestions/questionsDatasets_Small_OR.csv'

preproc = Preprocessor(train_csv1)  # 'E:/TFM/final/complete.csv')
# E:\TFM\Dataset1Car\train
nlp = spacy.load("en_core_web_sm")

# Get the dataset object
train_data = preproc.train_data

# Looking at the Vocabulary
print(preproc.prog_f.vocab.stoi)

# Looking at the Vocabulary
print(preproc.que_f.vocab.stoi)

# Training
# Training hyperparameters
num_epochs = 6
n_epochs = 10
learning_rate = 3e-4
batch_size = 300
num_steps = len(train_data) / batch_size

folder = "F:/Github/COCO NSVQA/Classificator/"

dataset_dir2 = "E:/TFM/Dataset1Car/images/"
dataset_dir = "E:/TFM/car/car_data/car_data/"

# Model hyperparameters
config = {
    'que_vocab_size': len(preproc.que_f.vocab),
    'prog_vocab_size': len(preproc.prog_f.vocab),
    'embedding_dim': 256 * 2,
    'num_heads': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'dropout': 0.10,
    'max_len': 800,
    'forward_expansion': 4,
    'que_pad_idx': preproc.que_f.vocab.stoi["<pad>"]
}

train_tfms = transforms.Compose([transforms.Resize((200, 200)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor()])
test_tfms = transforms.Compose([transforms.Resize((200, 200)),
                                transforms.ToTensor()])

from collections.abc import Iterable  # import directly from collections for Python < 3.3


class NSAIPipeline():
    '''End-to-End Pipeline of Neuro-Symbolic AI on Sort-of-CLEVR dataset'''

    def __init__(self,
                 config,
                 detector='models/detector.svm',
                 classifier='models/classifier.pth',
                 sem_parser='../Questions/semantic_parser.pth',
                 train_csv=train_csv1,
                 device=None):

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Perception Module
        logging.basicConfig(level=logging.DEBUG,
                            format="[%(asctime)s %(filename)s] %(message)s")

        if len(sys.argv) != 2:
            logging.error("Usage: python test_images.py params.py")
            sys.exit()
        params_path = sys.argv[1]
        if not os.path.isfile(params_path):
            logging.error("no params file found! path: {}".format(params_path))
            sys.exit()
        config2 = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
        config2["batch_size"] *= len(config2["parallels"])

        self.perceiver = Scene_Parser(config2)

        # Semantic Parser
        self.preproc = Preprocessor(train_csv)
        self.sem_parser = SemanticParser(self.preproc, config, filename=sem_parser, device=self.device)

        # Program Executor
        self.executor = ProgramExecutor()

    def predict(self, img, query):
        '''
        Make Prediction on a single image and question pair

        Args:
            img (str/array): pixel values should be in 0-255 range
                             of dtype uint8 in BGR color format or
                             file path of the image
            query (str): question about the image

        Returns:
            str: answer of the query
        '''
        # Load img if it's a path
        # if type(img) == str:
        #    img = cv2.imread(img)

        # Structured Scene Representation
        # scene = self.perceiver.scene_repr(img)
        # prepare images path
        # with open(
        #         'F:/Github/YOLOv3_PyTorchF/scene_parser/v2_OpenEnded_mscoco_train2014_questions_howmanyDATASET1.json',
        #         'r') as g:
        #     QAs = json.load(g)
        #
        # keysList = [key for key in QAs]
        # print(keysList)
        #
        # scene=[]
        # name = img.replace('COCO_train2014_', "")
        # name = name.replace('.jpg', "")
        # #scene = QAs.get(name)
        # for im in QAs:
        #     if int(name) == int(im):
        #         scene = QAs.get(im)

        # Synthesize Program from Query
        #img = "E:/TFM/Dataset1Car/trainMod/Screen-Shot-2022-04-14-at-12-30-26-AM_png.rf.c26e6e52a479f963e5c11bc64a63916b.jpg"
        program = self.sem_parser.predict(query)
        program2 = []
        for p in program:
            p = p.split(" ")
            if len(p) > 1:
                p1 = p[0] + " "
                for e in p[1:]:
                    p1 += e
                program2.append(p1)
            else:
                program2.append(p[0])

        program = program2
        # Execute Program
        print(program)

        scene = self.perceiver.predict(img)
        print(scene)

        answer = self.executor(scene, program)

        return answer, program

    def evaluateTypeX(self):
        for a in answers:
            if a.isalpha():
                s += a
            if a == ",":
                ans.append(s)
                s = ""
        ok = False
        for a in pred:
            if str(a) in ans:
                ok = True
        if "for-sale" in filename:
            name = filename.split("_")[1]
            name = name.replace("-for-sale", "")
            newModels.add(name.replace("-", " "))
        # Verify answer
        if not ok:
            if debug:
                print(filename, question, pred_prog, pred, answers)
            correct.append(0)
            bad += 1
        else:
            correct.append(1)
            good += 1

    def evaluateType1(self, pred, anwers, correct):

        if isinstance(anwers, str):
            anwers = (anwers == 'True')
        if pred == anwers:
            correct.append(1)
            return correct, 1, 0
        else:
            correct.append(0)
            return correct, 0, 1

    def evaluateTypeCount(self, pred, anwers, correct):
        if str(pred) == str(anwers):
            correct.append(1)
            return correct, 1, 0
        else:
            print("Pred:" + str(pred) + "-Ans:" + str(anwers))
            correct.append(0)
            return correct, 0, 1

    def evaluateTypeStr(self, pred, anwers, correct):
        if pred in anwers:
            correct.append(1)
            return correct, 1, 0
        else:

            print("Pred:" + str(pred) + "-Ans:" + str(anwers))
            correct.append(0)
            return correct, 0, 1

    def evaluateTypeObject(self, pred, anwers, correct):
        if pred == anwers:
            correct.append(1)
            return correct, 1, 0
        elif not anwers:
            correct.append(1)
            return correct, 1, 0
        else:
            print("Pred:" + str(pred) + "-Ans:" + str(anwers))
            correct.append(0)
            return correct, 0, 1


    def evaluate(self, csv, img_dir, debug=True):

        data = pd.read_csv(csv).values
        good = 0
        bad = 0
        correct = []

        q_1_good = 0
        q_2_good = 0

        q_1_bad = 0
        q_2_bad = 0
        log_text =""
        q1_correct = []
        q2_correct = []
        i = 0
        for filename, question, question_type, program, answers in tqdm(data):

            # Load Image
            img_path = img_dir + filename
            # img = cv2.imread(img_path)

            # Make prediction
            print(question)
            try:
                i += 1
                pred, pred_prog = self.predict(img_path, question)
                ans = []
                s = ""
                if isinstance(pred, bool):
                    correct, g, b = self.evaluateType1(pred, answers, correct)
                    if g:
                        print("Good Classif: " + question, " " + img_path)
                    if b:
                        print("Bad Classif: " + question, " " + img_path)
                        print("Answ: " + str(answers), " ,PRED: " + str(pred))
                        log_text += question + " - " + img_path+ " - " + "Answ: " + str(answers) + " "  + "PRED:"  + str(pred) + '\n'
                    good += g
                    bad += b

                    if question_type == 1:
                        q_1_good += g
                        q_1_bad += b
                    else:
                        q_2_good += g
                        q_2_bad += b
                elif isinstance(pred, int):
                    correct, g, b = self.evaluateTypeCount(pred, answers, correct)
                    if g :
                        print("Good Classif: " + question, " " + img_path)
                    if b :
                        print("Bad Classif: " + question, " " + img_path)
                        print("Answ: " + str(answers), " ,PRED: " + str(pred))
                        log_text += question + " - " + img_path+ " - " + "Answ: " + str(answers) + " "  + "PRED:"  + str(pred) + '\n'
                    good += g
                    bad += b

                    if question_type == 1:
                        q_1_good += g
                        q_1_bad += b
                    else:
                        q_2_good += g
                        q_2_bad += b
                elif isinstance(pred, str):
                    correct, g, b = self.evaluateTypeStr(pred, answers, correct)
                    if g:
                        print("Good Classif: " + question, " " + img_path)
                    if b :
                        print("Bad Classif: " + question, " " + img_path)
                        print("Answ: " + str(answers), " ,PRED: " + str(pred))
                        log_text += question + " - " + img_path + " - " + "Answ: " + str(answers) + " " + "PRED:" + str(
                            pred) + '\n'
                    good += g
                    bad += b

                    if question_type == 1:
                        q_1_good += g
                        q_1_bad += b
                    else:
                        q_2_good += g
                        q_2_bad += b
                elif isinstance(pred, list):
                    enc=False
                    for p in pred:
                        if isinstance(p, str):
                            correct, g, b = self.evaluateTypeStr(p, answers, correct)
                            if g:
                                enc=True
                                break
                        elif isinstance(p, dict):
                            if isinstance(answers, str):
                                answers = answers.replace('\\', "").replace('"', "").replace("'", "")
                                answers = answers.strip('][').split(', ')
                            for a in answers:
                                correct, g, b = self.evaluateTypeObject(p["class"], a, correct)
                                if g:
                                    enc=True
                                    break


                    if g :
                        print("Good Classif: " + question, " " + img_path)
                    if b :
                        print("Bad Classif: " + question, " " + img_path)
                        print("Answ: " + str(answers), " ,PRED: " + str(pred))
                        log_text += question + " - " + img_path+ " - " + "Answ: " + str(answers) + " "  + "PRED:"  + str(pred) + '\n'
                    if enc:
                        good += 1
                        bad += 0
                        if question_type == 1:
                            q_1_good += 1
                            q_1_bad += 0
                        else:
                            q_2_good += 1
                            q_2_bad += 0
                    else:
                        good += 0
                        bad += 1
                        if question_type == 1:
                            q_1_good += 0
                            q_1_bad += 1
                        else:
                            q_2_good += 0
                            q_2_bad += 1
                        log_text += question + " - " + img_path+ " - " + "Answ: " + str(answers) + " "  + "PRED:"  + str(pred) + '\n'
                        print("Answ: " + str(answers), " ,PRED: " + str(pred))


                print("Good: " + str(good) + " bad: " + str(bad))
                print("Q1 Good: " + str(q_1_good) + " bad: " + str(q_1_bad))
                print("Q2 Good: " + str(q_2_good) + " bad: " + str(q_2_bad))
                print(i)
            except Exception as e:
                logging.error(traceback.format_exc())

        acc = (sum(correct) / len(correct)) * 100

        print((good / i) * 100)
        with open('log.txt', 'a') as log_file:
            log_file.write(log_text + '\n')

        return acc

    def train(self):  ##Only training classification part

        dataset = torchvision.datasets.ImageFolder(root=dataset_dir + "train")
        trainloaderT = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.perceiver.model_classifier.model_ft.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold=0.9)

        losses = []
        accuracies = []
        test_accuracies = []
        # set the model to train mode initially
        self.perceiver.model_classifier.model_ft.train()
        for epoch in range(n_epochs):
            since = time.time()
            running_loss = 0.0
            running_correct = 0.0

            good = 0
            bad = 0
            correct = []
            for i, data in enumerate(trainloaderT, 0):

                inputs, labels = data
                # inputs = inputs.to(device).half() # uncomment for half precision model
                input = input.to(device)
                labels = labels.to(device)
                question = "What is the model of the car?"
                answers = labels

                # image = Image.open(img)
                # img = train_tfms(img).float()
                # # trans = T.ToPILImage()
                # # imgV = trans(img)
                # # imgV.show()
                # img = torch.autograd.Variable(img, requires_grad=True)
                # img = img.unsqueeze(0)
                # img = img.cuda()

                optimizer.zero_grad()
                outputs_array = self.perceiver.model_classifier.trainModel(input)

                for outputs in outputs_array:
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, answers)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    running_correct += (labels == predicted).sum().item()

                epoch_duration = time.time() - since
                epoch_loss = running_loss / len(trainloader)
                epoch_acc = 100 / 32 * running_correct / len(trainloader)
                print(
                    "Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (
                        epoch + 1, epoch_duration, epoch_loss, epoch_acc))

                losses.append(epoch_loss)
                accuracies.append(epoch_acc)

                # switch the model to eval mode to evaluate on test data
                self.perceiver.model_classifier.model_ft.eval()
                test_acc = eval_model(self.perceiver.model_classifier.model_ft)
                test_accuracies.append(test_acc)

                # re-set the model to train mode after validating
                self.perceiver.model_classifier.model_ft.train()
                scheduler.step(test_acc)
                since = time.time()

        torch.save(self.perceiver.model_classifier.model_ft.state_dict(),
                   'model_weightsTotal.pth')  # model.load_state_dict(torch.load('model_weights.pth'))


def eval_model(model):
    correct = 0.0
    total = 0.0
    dataset2 = torchvision.datasets.ImageFolder(root=dataset_dir + "test", transform=test_tfms)
    testloader = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=2)

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            # inputs = inputs.to(device).half() # uncomment for half precision model
            input = input.to(device)
            labels = labels.to(device)
            question = "What is the model of the car"
            answers = labels

            # image = Image.open(img)
            # img = train_tfms(img).float()
            # # trans = T.ToPILImage()
            # # imgV = trans(img)
            # # imgV.show()
            # img = torch.autograd.Variable(img, requires_grad=True)
            # img = img.unsqueeze(0)
            # img = img.cuda()

            optimizer.zero_grad()
            outputs_array = self.perceiver.model_classifier.trainModel(input)

            for outputs in outputs_array:
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            total += labels.size(0)

    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (
        test_acc))
    return test_acc


if __name__ == "__main__":
    nsai = NSAIPipeline(config)
    print(nsai.predict('E:/TFM/Dataset1Car/trainMod/64_jpg.rf.8ab486917b04d17e6beea8dca62e0e1f.jpg', 'What number of other objects are the same type as the person?'))
    #_id_0_maker_bentley_model_mulsanne_006dce266f.jpg
    # print(nsai.evaluate('E:/TFM/FGVD/questionsFGVDPT.csv', 'E:/TFM/IDD_FGVD/train/images/'))
    print(nsai.evaluate('E:/TFM/Dataset1Car/datasetQuestions/questionsDataset_Small_locationC.csv',
                        'E:/TFM/Dataset1Car/trainMod/'))
    # "E:/TFM/Dataset1Car/questionsDataset1.csv" questionsDataset_Small_location.csv'

    # nsai.train()
    # print(nsai.predict('E:/TFM/IDD_FGVD/train/images/4790.jpg', 'What model is the car?'))
    # print(nsai.predict('E:/TFM/IDD_FGVD/train/images/4790.jpg', 'What color is the car?'))
    # print(nsai.predict('E:/TFM/IDD_FGVD/train/images/4790.jpg', 'What is next to the car?'))
    # print(nsai.predict('E:/TFM/IDD_FGVD/train/images/4790.jpg', 'How many cars are there?'))
    # print(nsai.predict('E:/TFM/IDD_FGVD/train/images/4790.jpg', 'Is there a car?'))
    # print(nsai.predict('E:/TFM/IDD_FGVD/train/images/4790.jpg', 'What shows in the image?'))
    #
    # print(nsai.predict('E:/TFM/COCO/images/train2014/COCO_01.jpg', 'What model is the car?'))
    # print(nsai.predict('E:/TFM/COCO/images/train2014/COCO_01.jpg', 'What color is the car?'))
    # print(nsai.predict('E:/TFM/COCO/images/train2014/COCO_01.jpg', 'What is next to the car?'))
    # print(nsai.predict('E:/TFM/COCO/images/train2014/COCO_01.jpg', 'How many cars are there?'))
    # print(nsai.predict('E:/TFM/COCO/images/train2014/COCO_01.jpg', 'Is there a car?'))
    # print(nsai.predict('E:/TFM/COCO/images/train2014/COCO_01.jpg', 'What shows in the image?'))
    # print(nsai.predict('E:/TFM/COCO/images/train2014/COCO_01.jpg', 'What is behind the black car?'))

    # COCO_train2014_000000425864_models
    # How many cars are shown, Is there a bus?, 'Where is the bus?', Where is the bus on the right?

# 'Where is the bus on the left?': ([(150.33428955078125, 112.51100158691406)], ['filter bus', 'location left', 'query location'])

# 'What is the object on the right?': ('person', ['filter object', 'location right', 'query what'])
