import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import cm
import time
import os

from IPython.display import display
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


dataset_dir = "E:/TFM/color/"
classes, c_to_idx = find_classes(dataset_dir + "train")

folder="E:/Codigos TFM/COCO NSVQA/Classificator/"



def train_model(model, criterion, optimizer, scheduler, n_epochs=15):
    losses = []
    accuracies = []
    test_accuracies = []
    labelsF = set()
    # set the model to train mode initially
    model.train()
    for epoch in range(n_epochs):
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        all_predicted = []
        all_labels = []

        for i, data in enumerate(trainloader, 0):
            # get the inputs and assign them to cuda
            inputs, labels = data
            # inputs = inputs.to(device).half() # uncomment for half precision model
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate the loss/acc later
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

            # Collect predicted labels and true labels for confusion matrix
            all_predicted.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

        epoch_duration = time.time() - since
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 / 32 * running_correct / len(trainloader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch + 1, epoch_duration, epoch_loss, epoch_acc))

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        # switch the model to eval mode to evaluate on test data
        model.eval()
        test_acc = eval_model(model)
        test_accuracies.append(test_acc)

        # re-set the model to train mode after validating
        model.train()
        scheduler.step(test_acc)
        since = time.time()

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(all_predicted, all_labels)

    sns.set(font_scale=0.5)
    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, fmt='g',
                ax=ax)  # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


    # re-set the model to train mode after validating
    model.train()
    scheduler.step(test_acc)
    since = time.time()


    torch.save(model.state_dict(), 'model_weights_color2.pth')  # model.load_state_dict(torch.load('model_weights.pth'))

    print('Finished Training')
    return model, losses, accuracies, test_accuracies


def eval_model(model):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            # images = images.to(device).half() # uncomment for half precision model
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (
        test_acc))
    return test_acc


class Classifier():
    def __init__(self):
        self.model_ft = models.resnet34(pretrained=True)
        self.num_ftrs = self.model_ft.fc.in_features

        # replace the last fc layer with an untrained one (requires grad by default)
        self.model_ft.fc = nn.Linear(self.num_ftrs, 15)
        self.model_ft = self.model_ft.to(device)

        self.model_ft.load_state_dict(torch.load(folder + 'model_weights_color.pth'))
        self.model_ft.eval()
        self.classes, self.c_to_idx = find_classes(dataset_dir + "train")

    def predict(self, img):
        #img = Image.fromarray(img)
        

        img = train_tfms(img).float()
        img = torch.autograd.Variable(img, requires_grad=True)
        img = img.unsqueeze(0)
        img = img.cuda()
        
        #img = train_tfms(img)



        output = self.model_ft(img)

        conf, predicted = torch.max(output.data, 1)

        return self.classes[predicted.item()], conf.item()


train_tfms = transforms.Compose([transforms.Resize((200, 200)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor()])
test_tfms = transforms.Compose([transforms.Resize((200, 200)),
                                transforms.ToTensor()])


if __name__ == '__main__':
    dataset = torchvision.datasets.ImageFolder(root=dataset_dir + "train", transform=train_tfms)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    print(dataset.classes)
    dataset2 = torchvision.datasets.ImageFolder(root=dataset_dir + "test", transform=test_tfms)
    testloader = torch.utils.data.DataLoader(dataset2, batch_size=32, shuffle=False, num_workers=2)

    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    # replace the last fc layer with an untrained one (requires grad by default)
    model_ft.fc = nn.Linear(num_ftrs, 15)
    model_ft = model_ft.to(device)

    # uncomment this block for half precision model
    """
    model_ft = model_ft.half()


    for layer in model_ft.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

    """
    probably not the best metric to track, but we are tracking the training accuracy and measuring whether
    it increases by atleast 0.9 per epoch and if it hasn't increased by 0.9 reduce the lr by 0.1x.
    However in this model it did not benefit me.
    """
    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold=0.9)

    model_ft, training_losses, training_accs, test_accs = train_model(model_ft, criterion, optimizer, lrscheduler,
                                                                      n_epochs=15)

    # plot the stats

    f, axarr = plt.subplots(2, 2, figsize=(12, 8))
    axarr[0, 0].plot(training_losses)
    axarr[0, 0].set_title("Training loss")
    axarr[0, 1].plot(training_accs)
    axarr[0, 1].set_title("Training acc")
    axarr[1, 0].plot(test_accs)

    axarr[1, 0].set_title("Test acc")

    # tie the class indices to their names

    classes, c_to_idx = find_classes(dataset_dir + "train")

    # test the model on random images

    # switch the model to evaluation mode to make dropout and batch norm work in eval mode
    model_ft.eval()

    # transforms for the input image
    loader = transforms.Compose([transforms.Resize((200, 200)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open(dataset_dir + "test/pink/1b60b2d9b7.jpg")
    image = loader(image).float()
    image = torch.autograd.Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    image = image.cuda()
    output = model_ft(image)
    conf, predicted = torch.max(output.data, 1)

    # get the class name of the prediction
    display(Image.open(dataset_dir + "test/pink/1b60b2d9b7.jpg"))
    print(classes[predicted.item()], "confidence: ", conf.item())
