from bs4 import BeautifulSoup

dir_path = "E:/TFM/IDD_FGVD/train/annos/"

# Loop through all files in the directory
count = 0

Q1 = {}
Q2 = []
for filename in os.listdir(dir_path):
    if filename.endswith(".xml"):
        # Reading the data inside the xml
        # file to a variable under the name
        # data
        with open(filename + '.xml', 'r') as f:
            data = f.read()
        # Passing the stored data inside
        # the beautifulsoup parser, storing
        # the returned object
        Bs_data = BeautifulSoup(data, "xml")

        print(1)
