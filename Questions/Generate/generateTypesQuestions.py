import random
import json

rcEmpty={"size": "", "color": "", "shape": "", "type":""}


def randomObjects(obj, aComp, aColor, aModel, aType):
    typeO=random.choice(aType)
    if typeO=="car":
        objectC = {"company": random.choice(aComp) if aComp else "",
                   "color": random.choice(aColor) if aColor else "",
                   "model": random.choice(aModel)if aModel else "",
                   "type": typeO}
        while objectC == obj:
            objectC = {"company": random.choice(aComp) if aComp else "",
                   "color": random.choice(aColor) if aColor else "",
                   "model": random.choice(aModel)if aModel else "",
                   "type": random.choice(aType)}
    else:
        objectC = {"company":"",
                   "color": "",
                   "model": "",
                   "type": typeO}
        while objectC == obj:
            objectC = randomObjects(obj,aComp, aColor, aModel, aType)

    return objectC


##Compare Integer/Number of objects
def constructObject_compareIntegerQ_0(aComp, aColor, aModel,  obj1):
    result = ""
    if aComp:
        result +=  obj1.get("company") +" "
    if aColor:
        result += obj1.get("color") +" "
    if aModel:
        result += obj1.get("model") +" "
    result += obj1.get("type") 
    return result


def constructProgram_compareIntegerQ_0(aComp, aColor, aModel,  obj1):
    result = "newObjects"
    if  aComp:
        result += " <nxt> filter_company " + obj1.get("company")
    if  aColor:
        result += " <nxt> filter_color " + obj1.get("color")
    if  aModel:
        result += " <nxt> filter_model " + obj1.get("model")
    result += " <nxt> filter_type " + obj1.get("type")
    return result

#Construct Complete Q-P
def construct_compareIntegerQ_0(aComp, aColor, aModel, aType):
    obj1 = randomObjects(rcEmpty, aComp, aColor, aModel, aType)
    obj2 = randomObjects(obj1, aComp, aColor, aModel, aType)

    ##Complete Object Construction
    rQ1 = constructObject_compareIntegerQ_0(aComp, aColor, aModel, obj1)
    rQ2 = constructObject_compareIntegerQ_0(aComp, aColor, aModel, obj2)

    ##Complete Program Construction
    rP1 = constructProgram_compareIntegerQ_0(aComp, aColor, aModel, obj1)
    rP2 = constructProgram_compareIntegerQ_0(aComp, aColor, aModel, obj2)

    compareIntegerQ_1 = [
        "Are there an equal number of " + rQ1 + "s and " + rQ2 + "s?",
        "Are there the same number of " + rQ1 + "s and " + rQ2 + "s?",
    "Is the number of " + rQ1 + "s the same as the number of " + rQ2 + "s?"]

    completeQuestion = random.choice(compareIntegerQ_1)
    completeProgram = rP1 + " <nxt> count <nxt> " + rP2 + " <nxt> count <nxt> " + "is_equal"

    return (completeQuestion, completeProgram)

def construct_compareIntegerQ_1(aComp, aColor, aModel, aType):
    obj1 = randomObjects(rcEmpty, aComp, aColor, aModel, aType)
    obj2 = randomObjects(obj1, aComp, aColor, aModel, aType)

    ##Complete Object Construction
    rQ1 = constructObject_compareIntegerQ_0(aComp, aColor, aModel, obj1)
    rQ2 = constructObject_compareIntegerQ_0(aComp, aColor, aModel, obj2)

    ##Complete Program Construction
    rP1 = constructProgram_compareIntegerQ_0(aComp, aColor, aModel, obj1)
    rP2 = constructProgram_compareIntegerQ_0(aComp, aColor, aModel, obj2)

    compareIntegerQ_1 = [
         "Is the number of " + rQ1 + "s less than the number of " + rQ2 + "s?",
    "Are there fewer " + rQ1 + "s than objects " + rQ2 + "s?"]

    completeQuestion = random.choice(compareIntegerQ_1)
    completeProgram = rP1 + " <nxt> count <nxt> " + rP2 + " <nxt> count <nxt> " + "is_less"

    return (completeQuestion, completeProgram)

def construct_compareIntegerQ_2(aComp, aColor, aModel, aType):
    obj1 = randomObjects(rcEmpty, aComp, aColor, aModel, aType)
    obj2 = randomObjects(obj1, aComp, aColor, aModel, aType)

    ##Complete Object Construction
    rQ1 = constructObject_compareIntegerQ_0(aComp, aColor, aModel, obj1)
    rQ2 = constructObject_compareIntegerQ_0(aComp, aColor, aModel, obj2)

    ##Complete Program Construction
    rP1 = constructProgram_compareIntegerQ_0(aComp, aColor, aModel, obj1)
    rP2 = constructProgram_compareIntegerQ_0(aComp, aColor, aModel, obj2)
    compareIntegerQ_1 = [
         "Is the number of " + rQ1 + "s greater than the number of " + rQ2 + "s?",
    "Are there more " + rQ1 + "s than objects " + rQ2 + "s?"]

    completeQuestion = random.choice(compareIntegerQ_1)
    completeProgram = rP1 + " <nxt> count <nxt> " + rP2 + " <nxt> count <nxt> " + "is_more"

    return (completeQuestion, completeProgram)


##Only 1


def construct_singleQ_0(aComp, aColor, aModel, aType):
    obj1 = randomObjects(rcEmpty, aComp, aColor, aModel, aType)

    ##Complete Object Construction
    rQ1 = constructObject_compareIntegerQ_0(aComp, aColor, aModel, obj1)

    ##Complete Program Construction
    rP1 = constructProgram_compareIntegerQ_0(aComp, aColor, aModel, obj1)

    compareIntegerQ_1 = [
        "How many " + rQ1 + "s are there?",
        "What number of " + rQ1 + "s are there?"]

    completeQuestion = random.choice(compareIntegerQ_1)
    completeProgram = rP1 + " <nxt> count"

    return (completeQuestion, completeProgram)

def construct_singleQ_1(aComp, aColor, aModel, aType):
    obj1 = randomObjects(rcEmpty, aComp, aColor, aModel, aType)

    ##Complete Object Construction
    rQ1 = constructObject_compareIntegerQ_0(aComp, aColor, aModel, obj1)

    ##Complete Program Construction
    rP1 = constructProgram_compareIntegerQ_0(aComp, aColor, aModel, obj1)

    compareIntegerQ_1 = [
        "Are there any " + rQ1 + "s?",
        "Is there a " + rQ1 + "?",
        "Are any " + rQ1 + "s visible?"]

    completeQuestion = random.choice(compareIntegerQ_1)
    completeProgram = rP1 + " <nxt> exist"

    return (completeQuestion, completeProgram)

def construct_singleQ_Model(aComp, aColor, aModel, aType):
    obj1 = randomObjects(rcEmpty, aComp, aColor, [], aType)

    ##Complete Object Construction
    rQ1 = constructObject_compareIntegerQ_0(aComp, aColor, [], obj1)

    ##Complete Program Construction
    rP1 = constructProgram_compareIntegerQ_0(aComp, aColor, [], obj1)

    compareIntegerQ_1 = [
        "What model is the " + rQ1 + "?",
        "What is the model of the " + rQ1 + "?",
        "There is a " + rQ1 + "; what model is it?",
        "The " + rQ1 + " is what model?",
        "The " + rQ1 + " has what model"]

    completeQuestion = random.choice(compareIntegerQ_1)
    completeProgram = rP1 + " <nxt> query model"

    return (completeQuestion, completeProgram)

def construct_singleQ_Color(aComp, aColor, aModel, aType):
    obj1 = randomObjects(rcEmpty, aComp, [], aModel, aType)

    ##Complete Object Construction
    rQ1 = constructObject_compareIntegerQ_0(aComp, [], aModel, obj1)

    ##Complete Program Construction
    rP1 = constructProgram_compareIntegerQ_0(aComp, [], aModel, obj1)

    compareIntegerQ_1 = [
        "What color is the " + rQ1 + "?",
        "What is the color of the " + rQ1 + "?",
        "There is a " + rQ1 + "; what color is it?",
        "The " + rQ1 + " is what color?",
        "The " + rQ1 + " has what color"]

    completeQuestion = random.choice(compareIntegerQ_1)
    completeProgram = rP1 + " <nxt> query color"

    return (completeQuestion, completeProgram)

def construct_singleQ_Company(aComp, aColor, aModel, aType):
    obj1 = randomObjects(rcEmpty, [], aColor, [], aType)

    ##Complete Object Construction
    rQ1 = constructObject_compareIntegerQ_0([], aColor, [], obj1)

    ##Complete Program Construction
    rP1 = constructProgram_compareIntegerQ_0([], aColor, [], obj1)

    compareIntegerQ_1 = [
        "What company is the " + rQ1 + "?",
        "What is the company of the " + rQ1 + "?",
        "There is a " + rQ1 + "; what company is it?",
        "The " + rQ1 + " is what company?",
        "The " + rQ1 + " has what company", 
        "What manufacter is the " + rQ1 + "?",
        "What is the manufacter of the " + rQ1 + "?",
        "There is a " + rQ1 + "; what manufacter is it?",
        "The " + rQ1 + " is what manufacter?",
        "The " + rQ1 + " has what manufacter"]

    completeQuestion = random.choice(compareIntegerQ_1)
    completeProgram = rP1 + " <nxt> query company"

    return (completeQuestion, completeProgram)

def construct_singleQ_Location(aComp, aColor, aModel, aType):
    obj1 = randomObjects(rcEmpty, aComp, aColor, aModel, aType)

    ##Complete Object Construction
    rQ1 = constructObject_compareIntegerQ_0(aComp, aColor, aModel, obj1)

    ##Complete Program Construction
    rP1 = constructProgram_compareIntegerQ_0(aComp, aColor, aModel, obj1)

    compareIntegerQ_1 = [
        "What is next to the " + rQ1 + "?",
        "What is behind the " + rQ1 + "?",
        "What is behind the " + rQ1 + "s?",
        "What is in front of the " + rQ1 + "?",
        "What is in front of the " + rQ1 + "s?",
        "What is beside of the " + rQ1 + "?",
        "What is beside of the " + rQ1 + "s?",
        "What is on the " + rQ1 + "?",
        "What is near the " + rQ1 + "?",
        "What is by the " + rQ1 + "?",
        "What is next to the " + rQ1 + "s?"]

    completeQuestion = random.choice(compareIntegerQ_1)

    if completeQuestion.split(" ")[2]=="next":
        completeProgram = rP1 + " <nxt> location near"
    elif completeQuestion.split(" ")[2]=="beside":
        completeProgram = rP1 + " <nxt> location near"
    elif completeQuestion.split(" ")[2]=="near":
        completeProgram = rP1 + " <nxt> location near"
    elif completeQuestion.split(" ")[2]=="by":
        completeProgram = rP1 + " <nxt> location near"
    elif completeQuestion.split(" ")[2]=="behind":
        completeProgram = rP1 + " <nxt> location up"
    elif completeQuestion.split(" ")[2]=="in":
        completeProgram = rP1 + " <nxt> location down"
    elif completeQuestion.split(" ")[2]=="on":
        completeProgram = rP1 + " <nxt> location up"
    else:
        completeProgram = rP1 + " <nxt> location near"
    
    
    return (completeQuestion, completeProgram)

def construct_simple():
 

    compareIntegerQ_1 = [
        " What is in the picture?",
        "What is in the photo?",
        "What is in the photograph?",
        "What can you see in the picture?",
        "What can you see in the image?",
        "What can you see in the photo?",
        "What objects are visible in the photograph?",
        "What objects are visible in the image?",
        "What objects are visible in the picture?",
        "What objects are visible in the photo?",
        "What are the elements present in the picture?",
        "What are the elements present in the image?",
        "What are the elements present in the photograph?",
        "What are the elements present in the photo?",
        "What is in the image?"]

    completeQuestion = random.choice(compareIntegerQ_1)
    completeProgram = "query what"

    return (completeQuestion, completeProgram)

aComp=["Toyota"]
aColor=["black"]
aModel=["Modelo 1", "Modelo 2"]
aType =["car"]

#print(construct_compareIntegerQ_0(aComp, aColor, aModel, aType))

# with open('E:/TFM/myDataset/questions.json',
#           'w') as f:
#     json.dump(question, f)

# Print the question
