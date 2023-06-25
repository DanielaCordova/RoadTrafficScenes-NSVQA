import random

# Lists of words for each area of questioning
relation_words = ['next to', 'behind', 'in front of', 'beside']
location_words = ['on', 'at', 'near', 'by']
count_words = ['how many', 'number of', 'count of']
color_words = ['what color is', 'color of']
exists_words = ['does it have', 'is there', 'are there']

# List of vehicles and roads to ask about
vehicles = ['car', 'truck', 'bus', 'motorcycle']
roads = ['highway', 'street', 'road', 'freeway']

# Generate a random question based on a random area and object
area = random.choice([relation_words, location_words, count_words, color_words, exists_words])
object = random.choice([vehicles, roads])
question = f"{random.choice(area)} the {random.choice(object)}"

question=[]
for a in [relation_words, location_words, count_words, color_words, exists_words]:
    for elem in a:
        for ob in [vehicles, roads]:
            for o in ob:
                question = f"{elem} the {o}"


    with open(
            'E:/TFM/COCO/data/myDataset/questions.json',
            'w') as f:
        json.dump(QA2, f)

# Print the question
print(question)
