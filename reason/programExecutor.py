import numpy as np
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_md')

model_labels = ['AM General Hummer SUV 2000', 'Acura Integra Type R 2001', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008', 'Acura TSX Sedan 2012', 'Acura ZDX Hatchback 2012', 'Aston Martin V8 Vantage Convertible 2012', 'Aston Martin V8 Vantage Coupe 2012', 'Aston Martin Virage Convertible 2012', 'Aston Martin Virage Coupe 2012', 'Audi 100 Sedan 1994', 'Audi 100 Wagon 1994', 'Audi A5 Coupe 2012', 'Audi R8 Coupe 2012', 'Audi RS 4 Convertible 2008', 'Audi S4 Sedan 2007', 'Audi S4 Sedan 2012', 'Audi S5 Convertible 2012', 'Audi S5 Coupe 2012', 'Audi S6 Sedan 2011', 'Audi TT Hatchback 2011', 'Audi TT RS Coupe 2012', 'Audi TTS Coupe 2012', 'Audi V8 Sedan 1994', 'BMW 1 Series Convertible 2012', 'BMW 1 Series Coupe 2012', 'BMW 3 Series Sedan 2012', 'BMW 3 Series Wagon 2012', 'BMW 6 Series Convertible 2007', 'BMW ActiveHybrid 5 Sedan 2012', 'BMW M3 Coupe 2012', 'BMW M5 Sedan 2010', 'BMW M6 Convertible 2010', 'BMW X3 SUV 2012', 'BMW X5 SUV 2007', 'BMW X6 SUV 2012', 'BMW Z4 Convertible 2012', 'Bentley Arnage Sedan 2009', 'Bentley Continental Flying Spur Sedan 2007', 'Bentley Continental GT Coupe 2007', 'Bentley Continental GT Coupe 2012', 'Bentley Continental Supersports Conv. Convertible 2012', 'Bentley Mulsanne Sedan 2011', 'Bugatti Veyron 16.4 Convertible 2009', 'Bugatti Veyron 16.4 Coupe 2009', 'Buick Enclave SUV 2012', 'Buick Rainier SUV 2007', 'Buick Regal GS 2012', 'Buick Verano Sedan 2012', 'Cadillac CTS-V Sedan 2012', 'Cadillac Escalade EXT Crew Cab 2007', 'Cadillac SRX SUV 2012', 'Chevrolet Avalanche Crew Cab 2012', 'Chevrolet Camaro Convertible 2012', 'Chevrolet Cobalt SS 2010', 'Chevrolet Corvette Convertible 2012', 'Chevrolet Corvette Ron Fellows Edition Z06 2007', 'Chevrolet Corvette ZR1 2012', 'Chevrolet Express Cargo Van 2007', 'Chevrolet Express Van 2007', 'Chevrolet HHR SS 2010', 'Chevrolet Impala Sedan 2007', 'Chevrolet Malibu Hybrid Sedan 2010', 'Chevrolet Malibu Sedan 2007', 'Chevrolet Monte Carlo Coupe 2007', 'Chevrolet Silverado 1500 Classic Extended Cab 2007', 'Chevrolet Silverado 1500 Extended Cab 2012', 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 'Chevrolet Silverado 1500 Regular Cab 2012', 'Chevrolet Silverado 2500HD Regular Cab 2012', 'Chevrolet Sonic Sedan 2012', 'Chevrolet Tahoe Hybrid SUV 2012', 'Chevrolet TrailBlazer SS 2009', 'Chevrolet Traverse SUV 2012', 'Chrysler 300 SRT-8 2010', 'Chrysler Aspen SUV 2009', 'Chrysler Crossfire Convertible 2008', 'Chrysler PT Cruiser Convertible 2008', 'Chrysler Sebring Convertible 2010', 'Chrysler Town and Country Minivan 2012', 'Daewoo Nubira Wagon 2002', 'Dodge Caliber Wagon 2007', 'Dodge Caliber Wagon 2012', 'Dodge Caravan Minivan 1997', 'Dodge Challenger SRT8 2011', 'Dodge Charger SRT-8 2009', 'Dodge Charger Sedan 2012', 'Dodge Dakota Club Cab 2007', 'Dodge Dakota Crew Cab 2010', 'Dodge Durango SUV 2007', 'Dodge Durango SUV 2012', 'Dodge Journey SUV 2012', 'Dodge Magnum Wagon 2008', 'Dodge Ram Pickup 3500 Crew Cab 2010', 'Dodge Ram Pickup 3500 Quad Cab 2009', 'Dodge Sprinter Cargo Van 2009', 'Eagle Talon Hatchback 1998', 'FIAT 500 Abarth 2012', 'FIAT 500 Convertible 2012', 'Ferrari 458 Italia Convertible 2012', 'Ferrari 458 Italia Coupe 2012', 'Ferrari California Convertible 2012', 'Ferrari FF Coupe 2012', 'Fisker Karma Sedan 2012', 'Ford E-Series Wagon Van 2012', 'Ford Edge SUV 2012', 'Ford Expedition EL SUV 2009', 'Ford F-150 Regular Cab 2007', 'Ford F-150 Regular Cab 2012', 'Ford F-450 Super Duty Crew Cab 2012', 'Ford Fiesta Sedan 2012', 'Ford Focus Sedan 2007', 'Ford Freestar Minivan 2007', 'Ford GT Coupe 2006', 'Ford Mustang Convertible 2007', 'Ford Ranger SuperCab 2011', 'GMC Acadia SUV 2012', 'GMC Canyon Extended Cab 2012', 'GMC Savana Van 2012', 'GMC Terrain SUV 2012', 'GMC Yukon Hybrid SUV 2012', 'Geo Metro Convertible 1993', 'HUMMER H2 SUT Crew Cab 2009', 'HUMMER H3T Crew Cab 2010', 'Honda Accord Coupe 2012', 'Honda Accord Sedan 2012', 'Honda Odyssey Minivan 2007', 'Honda Odyssey Minivan 2012', 'Hyundai Accent Sedan 2012', 'Hyundai Azera Sedan 2012', 'Hyundai Elantra Sedan 2007', 'Hyundai Elantra Touring Hatchback 2012', 'Hyundai Genesis Sedan 2012', 'Hyundai Santa Fe SUV 2012', 'Hyundai Sonata Hybrid Sedan 2012', 'Hyundai Sonata Sedan 2012', 'Hyundai Tucson SUV 2012', 'Hyundai Veloster Hatchback 2012', 'Hyundai Veracruz SUV 2012', 'Infiniti G Coupe IPL 2012', 'Infiniti QX56 SUV 2011', 'Isuzu Ascender SUV 2008', 'Jaguar XK XKR 2012', 'Jeep Compass SUV 2012', 'Jeep Grand Cherokee SUV 2012', 'Jeep Liberty SUV 2012', 'Jeep Patriot SUV 2012', 'Jeep Wrangler SUV 2012', 'Lamborghini Aventador Coupe 2012', 'Lamborghini Diablo Coupe 2001', 'Lamborghini Gallardo LP 570-4 Superleggera 2012', 'Lamborghini Reventon Coupe 2008', 'Land Rover LR2 SUV 2012', 'Land Rover Range Rover SUV 2012', 'Lincoln Town Car Sedan 2011', 'MINI Cooper Roadster Convertible 2012', 'Maybach Landaulet Convertible 2012', 'Mazda Tribute SUV 2011', 'McLaren MP4-12C Coupe 2012', 'Mercedes-Benz 300-Class Convertible 1993', 'Mercedes-Benz C-Class Sedan 2012', 'Mercedes-Benz E-Class Sedan 2012', 'Mercedes-Benz S-Class Sedan 2012', 'Mercedes-Benz SL-Class Coupe 2009', 'Mercedes-Benz Sprinter Van 2012', 'Mitsubishi Lancer Sedan 2012', 'Nissan 240SX Coupe 1998', 'Nissan Juke Hatchback 2012', 'Nissan Leaf Hatchback 2012', 'Nissan NV Passenger Van 2012', 'Plymouth Neon Coupe 1999', 'Porsche Panamera Sedan 2012', 'Ram C-V Cargo Van Minivan 2012', 'Rolls-Royce Ghost Sedan 2012', 'Rolls-Royce Phantom Drophead Coupe Convertible 2012', 'Rolls-Royce Phantom Sedan 2012', 'Scion xD Hatchback 2012', 'Spyker C8 Convertible 2009', 'Spyker C8 Coupe 2009', 'Suzuki Aerio Sedan 2007', 'Suzuki Kizashi Sedan 2012', 'Suzuki SX4 Hatchback 2012', 'Suzuki SX4 Sedan 2012', 'Tesla Model S Sedan 2012', 'Toyota 4Runner SUV 2012', 'Toyota Camry Sedan 2012', 'Toyota Corolla Sedan 2012', 
                'Toyota Sequoia SUV 2012', 'Toyota Vitz KSP 130 2017', 'Volkswagen Beetle Hatchback 2012', 'Volkswagen Golf Hatchback 1991', 'Volkswagen Golf Hatchback 2012', 'Volvo 240 Sedan 1993', 'Volvo C30 Hatchback 2012', 'Volvo XC90 SUV 2007', 'smart fortwo Convertible 2012']

color_labels=['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']

company_labels = ['lincoln', 'nissan', 'plymouth', 'buick', 'infiniti', 'daewoo', 'dodge', 'ferrari', 'am', 'rolls-royce', 'isuzu', 'suzuki', 'land', 'cadillac', 'jaguar', 'audi', 'spyker', 'aston', 'honda', 'fiat', 'fisker', 'ram', 'volvo', 'mitsubishi', 'porsche', 'jeep', 'bmw', 'ford', 'mercedes-benz', 'smart', 'mini', 'chrysler', 'maybach', 'mclaren', 'bugatti', 'eagle', 'bentley', 'chevrolet', 'hummer', 'tesla', 'volkswagen', 'scion', 'hyundai', 'lamborghini', 'acura', 'mazda', 'toyota', 'geo', 'gmc']

import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def find_closest_object(object_a, other_objects):
    closest_object = None
    min_distance = float('inf')

    for obj in other_objects:
        distance = calculate_distance(object_a['x1'], object_a['y1'], obj['x1'], obj['y1'])
        if distance < min_distance:
            min_distance = distance
            closest_object = obj

    return closest_object

def is_rect_intersecting_another_rect(x1, y1, x2, y2, x3, y3, x4, y4):
    if (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4):
        return False
    else:
        return True



def similar_word(word, vocab):
    doc = nlp(word)
    doc2 = nlp(' '.join(vocab))
    sim = 0.2
    newToken = doc[0]
    if doc[0].lower_ in vocab:
        return doc[0].lower_
    if doc[0].lemma_ not in vocab or doc[0].lower_ not in vocab:
        for idx2, token2 in enumerate(doc2):
            if doc[0].similarity(token2) > sim:
                newToken = token2
                sim = doc[0].similarity(token2)
    return newToken.lemma_


COCO_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'vehicle', 'object']


class DSL():
    '''Domain Specific Language consisting of functions and relations'''

    def __init__(self):

        self.objN=[]
        # Value to Attribute Converter
        self.val2attr = {
            'class': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                      'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                      'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear', 'hair drier', 'toothbrush']}
        self.get_attr = lambda x: [k for k, v in self.val2attr.items() if x in v][0]

        # String to Function
        self.str2func = {'filter': self.filter_,
                         'query': self.query,
                         'count': self.count,
                         'relate': self.relate,
                         'isleft': self.isLeft,
                         'istop': self.isTop,
                         'exist': self.exist,
                         'location': self.location,
                         'filter_color': self.filter_color,
                         'filter_model': self.filter_model,
                         'filter_company': self.filter_company,
                         'filter_type': self.filter_type,
                         'query_company': self.query_company,
                         'query_model': self.query_model,
                         'query_type': self.query_type,
                         'is_less': self.is_less,
                         'is_more': self.is_more,
                         'is_equal': self.is_equal,
                         'create_param': self.create_param,
                         'more_than_one': self.more_than_one,
                         'reduce_by_one': self.reduce_by_one,
                         'union': self.union,
                         'query_what': self.query_what,
                         'location_near': self.location_near}

    def more_than_one(self, obj):
        return obj>1

    def reduce_by_one(self, obj):
        return obj-1

    def sum_Vehicles(self):
        o = []
        for v in ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']:
            o = self.filter_(v,o)
        return o

    def sum_Vehicles2(self, obj):
        o = []
        for v in ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']:
            o += self.filter_type(obj,v)
        return o

    def filter_(self, param, obj):
        '''Returns a subset of the scene based on the param'''
        # Filter Object(s) for scene
        # attr = self.get_attr(param)
        param = similar_word(param, COCO_labels)

        filtered_objects = []

        if param == "vehicle":
            return self.sum_Vehicles()
        elif param == "object":
            if len(obj)>0:
                return obj
            else:
                return self.scene

        for im in self.scene:
            if (im.get('class') == param) or similar_word(im.get('class'), COCO_labels)==param:
                filtered_objects.append(im)
        # filtered_objects = self.scene[self.scene[attr] == param]
        if len(filtered_objects)==0:
            filtered_objects=obj
        elif obj is list:
            if len(obj)>0 :
                filtered_objects2=[]
                for o in obj:
                    if o in filtered_objects:
                        filtered_objects2.append(o)
                if len(filtered_objects2)==0:
                    filtered_objects2+=obj
                    filtered_objects2 += filtered_objects
                return filtered_objects2

        return filtered_objects

    def query(self, obj, attr):
        '''Returns the column value of object(s)'''
        result = []
        # i = obj.index[0]
        # result = obj[attr][i]
        if len(obj) > 0:
            if (attr == 'location'):
                result = [(o['x1'], o['y1']) for o in obj]
            elif(attr == 'what'):
                # for e in self.scene:
                #     if len(obj)>0:
                #         for o in obj:
                #             inters = is_rect_intersecting_another_rect(e['x1'], e['y1'], e['w'], e['h'], o['x1'], o['y1'], o['w'], o['h'])
                #             if inters:
                #                 result.append(e)
                #     else:
                #         result=self.scene
                result = obj
            elif (attr == 'color'):
                for o in obj:
                        result.append(o['color'])
            elif (attr == 'model'):
                for o in obj:
                        result.append(o['model'])
            elif (attr == 'manufacter' or attr == 'company'):
                for o in obj:
                        result.append(o['company'])
            else:
                result = obj[0]['class']
        else:
            result = [] #= self.scene

        return result

    def query_what(self, obj):
        return obj

    def query_company(self, obj):
        '''Returns the column value of object(s)'''
        result = []
        if len(obj) > 0:
            for o in obj:
                if o['class']=="car":
                    result.append(o['company'])
        else:
            result = [] #= self.scene

        return result

    def query_model(self, obj):
        '''Returns the column value of object(s)'''
        result = []
        if len(obj) > 0:
            for o in obj:
                if o['class']=="car":
                    result.append(o['model'])
        else:
            result = [] #= self.scene

        return result

    def query_type(self, obj):
        '''Returns the column value of object(s)'''
        result = set()
        if len(obj) > 0:
            for o in obj:
                    result.add(o['class'])
        else:
            result = [] #= self.scene

        return list(result)

    def location(self, obj, attr):
        '''Returns the column value of object(s)'''
        result = None
        attr = similar_word(attr, ['right', 'left', 'up', 'down'])
        if attr == 'right':
            x1 = 0
            for o in obj:
                if o['x1'] > x1:
                    result = o
                    x1 = o['x1']
        elif attr == 'left':
            x1 = float('inf')
            for o in obj:
                if o['x1'] < x1:
                    result = o
                    x1 = o['x1']
        elif attr == 'up':
            y1 = float('inf')
            for o in obj:
                if o['y1'] < y1:
                    result = o
                    y1 = o['y1']
        elif attr == 'down':
            y1 = 0
            for o in obj:
                if o['y1'] > y1:
                    result = o
                    y1 = o['y1']

        if result is None:
            return []
        return [result]



    def location_near(self, obj):
        '''Returns the column value of object(s)'''
        result = None
        completeScene=[]
        for e in self.scene:
            if e not in obj:
                completeScene.append(e)

        closest_object=[]
        for ob in obj:
            closest_object.append(find_closest_object(ob, completeScene))
        if closest_object is not None:
            result = closest_object
        if result is None:
            return []
        return result

    def relate(self, obj, param):
        '''Returns object which is either closest or furthest from all other objects of the scene'''
        obj_pos = self.query(obj, 'position')
        scene_pos = self.scene['position']

        # Calculate distances
        distances = np.array([np.linalg.norm(np.array(obj_pos) - np.array(pos)) for pos in scene_pos])

        sorted_dists = distances.argsort()

        if param == 'closest':
            idx = sorted_dists[0] if distances[sorted_dists[0]] != 0 else sorted_dists[1]
        elif param == 'furthest':
            idx = sorted_dists[-1] if distances[sorted_dists[-1]] != 0 else sorted_dists[-2]

        #         print(sorted_dists, distances, idx)

        # Get the object from the scene of that index
        req_obj = self.scene[self.scene.index == idx]

        return req_obj

    def count(self, objects):
        # '''Counts the objects'''
        # objects2=[]
        # if param!="" :
        #     param = similar_word(param)
        #     if param not in COCO_labels:
        #         return[]
        #     else:
        #         qa = self.filter_(param)
        # else:
        #     qa=[]
        #     for im in objects:
        #         if im not in qa:
        #             qa.append(im)

        return len(objects)

    def exist(self, objects):
        return len(objects) > 0

    def union(self, obj,param):
        return obj + param

    def is_equal(self, count_prev):
        return self.objN[-1]==count_prev

    def is_less(self, count_prev):
        return self.objN[-1]<count_prev

    def is_more(self, count_prev):
        return self.objN[-1]>count_prev

    def isLeft(self, pos):
        '''Checks if a position is on the left half or not'''
        return 'yes' if pos[0] < 112 else 'no'

    def isTop(self, pos):
        '''Checks if a position is on the top half or not'''
        return 'yes' if pos[1] < 112 else 'no'

    def createObject(self, prevScene):
        self.objN.append(prevScene)
        return self.scene

    def filter_type(self, obj, param):

        param = similar_word(param, COCO_labels)
        filtered_objects = []

        for im in obj:
            if (im.get('class') == param) or similar_word(im.get('class'), COCO_labels) == param:
                filtered_objects.append(im)
            if (im.get('class') == "truck") and param == "car":
                filtered_objects.append(im)

        if param == "vehicle":
            return self.sum_Vehicles2(obj)
        elif param == "object":
            if len(obj)>0:
                return obj
            else:
                return self.scene
        return filtered_objects

    def filter_model(self, obj, param):

        filtered_objects = []

        for im in obj:
            if im.get("class") == "car":
                if (im.get('model') == param) or similar_word(im.get('model'), model_labels) == param:
                    filtered_objects.append(im)

        return filtered_objects

    def filter_color(self, obj, param):

        filtered_objects = []

        for im in obj:
            if (im.get('color') == param) or similar_word(im.get('color'), color_labels) == param:
                filtered_objects.append(im)

        return filtered_objects

    def filter_company(self, obj, param):

        filtered_objects = []

        for im in obj:
            if im.get("class")=="car":
                if (im.get('company') == param) or similar_word(im.get('company'), company_labels) == param:
                    filtered_objects.append(im)

        return filtered_objects
    
    def create_param(self):
        return self.objN[-1]

class ProgramExecutor(DSL):
    '''Executes a given program'''

    def __init__(self):
        super().__init__()
        self.param=""
        pass

    def func_executor(self, func, param, prev_out):
        '''Executes a given function with or without a parameter'''
        # 0-1 arg functions

        if self.param!="":
            param = self.param

        if func in ['filter']:
            if param != None:
                prev_out = self.filter_(param, prev_out)
            else:
                self.filter_("", prev_out)           
        # Two arg functions
        elif func in ['query', 'relate', 'location', 'filter_type', 'filter_model', 'filter_color', 'filter_company', 'union']:
            if isinstance(param, list):
                for p in param:
                    prev_out = self.str2func[func](prev_out, p)
            else:
                prev_out = self.str2func[func](prev_out, param)

        # One arg functions
        elif func in ['count', 'isleft', 'istop', 'exist',  'query_company', 'query_model', 'is_equal', 'is_more', 'is_less', 'query_type',
                      'more_than_one', 'reduce_by_one', 'location_near']:
            prev_out = self.str2func[func](prev_out)

        elif func in ['newobjects']:
            prev_out = self.createObject(prev_out)

        if func in ['create_param']:
            self.param = self.str2func[func]()
        else:
            self.param=""

        return prev_out

    def __call__(self, scene, program):
        '''Executes a program on the scene'''
        self.scene = scene

        prev_out = []
        for seq in program:
            args = seq.split()
            # print(args)
            #             try:
            if len(args) < 2:
                prev_out = self.func_executor(args[0], None, prev_out)
            else:
                prev_out = self.func_executor(args[0], args[1], prev_out)
            # print(prev_out, '\n')
        #             except:
        #                 prev_out = 'Failed'

        return prev_out
