import os
import time
import io
import hashlib
import signal
import requests
from PIL import Image
from selenium import webdriver
import pandas as pd

import json
import argparse
import numpy as np
from webdriver_manager.chrome import ChromeDriverManager
import validators

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)

"""
    Credit:
    https://github.com/Ladvien/deep_arcane/blob/main/1_get_images/scrap.py

    Note: 
    Requires chromedriver installed in PATH variable

"""


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def fetch_image_urls(query: str, number_images: int, wd: webdriver, existing_urls: list,
                     sleep_between_interactions: float = 0.1) -> set:
    """

    :param query: str, individual query to search
    :param number_images: int, number of images to search for
    :param wd: selenium.webdriver.chrome.webdriver.WebDriver
    :param existing_urls: list, previously-download image links
    :param sleep_between_interactions: float, patience parameter
    :return: set, image URLs
    """

    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    start = time.time()
    while image_count < number_images:

        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(
            f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}"
        )

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract valid image urls
            actual_images = wd.find_elements_by_css_selector("img.pT0Scc")
            for actual_image in actual_images:
                if actual_image.get_attribute("src") and "http" in actual_image.get_attribute("src") and \
                        actual_image.get_attribute("src") not in existing_urls and validators.url(
                    actual_image.get_attribute("src")) == True:
                    image_urls.add(actual_image.get_attribute("src"))

            image_count = len(image_urls)

            if len(image_urls) >= number_images:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(20)

            if (time.time() - start) / 60 > 5:  # if still searching for >5 min, break and return whatever have
                break

            not_what_you_want_button = ""
            try:
                not_what_you_want_button = wd.find_element_by_css_selector(".r0zKGf")
            except:
                pass

            # If there are no more images return.
            if not_what_you_want_button:
                print("No more images available.")
                return image_urls

            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button and not not_what_you_want_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls


def search_and_download(wd: webdriver, query: str, root_dir_path: str, make_model_year: str,
                        number_images: int = 100) -> None:
    """
    Performs web search for an image query and downloads resulting images
    :param wd: selenium.webdriver.chrome.webdriver.WebDriver
    :param query: str, individual query to search
    :param root_dir_path: str, root directory path
    :param make_model_year: str, make-model-year
    :param number_images: int, number of images to search for
    :return: None
    """

    # Open JSON of image source URLs, if exists already, otherwise initialize
    if os.path.exists('E:/TFM/Dataset1Car/results/image_sources.json'):

        with open('E:/TFM/Dataset1Car/results/image_sources.json', 'rb') as j:
            existing_urls = json.load(j)

    else:
        existing_urls = {}

    res = fetch_image_urls(
        query,
        number_images,
        wd=wd,
        existing_urls=list(set(existing_urls.values()))
    )

    if res is not None:

        for url in res:

            ###### Download image ######
            try:
                print("Getting image")

                image_content = requests.get(url, verify=True,  timeout=5).content
                time.sleep(1)

            except Exception as e:
                print(f"ERROR - Could not download {url} - {e}")
                continue

            ##### Save image #####
            try:
                image_file = io.BytesIO(image_content)
                image = Image.open(image_file).convert("RGB")
                img_name = hashlib.sha1(image_content).hexdigest()[:10] + ".jpg"
                file_path = os.path.join(root_dir_path, make_model_year, img_name)
                with open(file_path, "wb") as f:
                    image.save(f, "JPEG", quality=85)
                print(f"SUCCESS - saved {url} - as {file_path}")

                # Add URL to successfully-saved image
                existing_urls[os.path.join(make_model_year, img_name)] = url  # only relative path to image

            except Exception as e:
                print(f"ERROR - Could not save {url} - {e}")

        with open('E:/TFM/Dataset1Car/results/image_sources.json', 'w') as j:
            json.dump(existing_urls, j)

    else:
        print(f"Failed to return links for term: {query}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default="E:/TFM/Dataset1Car/images/train/", help='path to output scraped images')
    parser.add_argument('--num-images', type=str, default=200,
                        help='number of images per detailed make-model class to scrape')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--top', action='store_true', default="--top", help='sort df ascending, begin with vehicle makes a -> z')
    group.add_argument('--bottom', action='store_true',help='sort df descending, begin with vehicle makes z -> a')
    return parser.parse_args()


def main(opt):
    # Read in database of makes and models to scrape
    # df = pd.read_csv('E:/TFM/Dataset1Car/make_model_database_modMerge.csv')
    #
    # # Remove vehicle make-model-year rows if dir already exists on disk (in case successfully ran previously)
    # lst = []
    # for subdir, dirs, files in os.walk(opt.output_path):
    #     for file in [i for i in files if 'jpg' in i or 'png' in i]:
    #         lst.append(subdir + '/' + file)  # does not count empty subdirectories
    # foo = pd.DataFrame(lst, columns=["Path"])
    # foo['Make'] = foo['Path'].apply(lambda x: x.split('/')[5].split(' ')[0])
    # foo['Model'] = foo['Path'].apply(lambda x: x.split('/')[5].split(' ')[1])
    # foo['Year'] = foo['Path'].apply(lambda x: x.split('/')[5].split(' ')[-1])
    # foo['Year'] = pd.to_numeric(foo['Year'], errors='coerce', downcast='integer').fillna(0).astype(int)
    # foo['dir'] = foo['Path'].apply(lambda x: '/'.join(x.split('/')[:-1]))
    #
    # # Fixes to account for Chevrolet C/K and RAM C/V
    # # Note - this was run on a MacBook. macOS behavior in Python changes '/' in strings is to ':'
    # foo.loc[(foo.Make == 'Chevrolet') & (foo.Model == 'C:K'), 'Model'] = 'C/K'
    # foo['dir'] = np.where((foo['Make'] == 'Chevrolet') & (foo['Model'] == 'C/K'),
    #                       'Chevrolet/C\/K/' + foo['Year'].astype(str), foo['dir'])
    # foo['Path'] = np.where((foo['Make'] == 'Chevrolet') & (foo['Model'] == 'C/K'),
    #                        foo['dir'] + '/' + foo['Path'].apply(lambda x: x.split('/')[-1]), foo['Path'])
    #
    # foo.loc[(foo.Make == 'RAM') & (foo.Model == 'C:V'), 'Model'] = 'C/V'
    # foo['dir'] = np.where((foo['Make'] == 'RAM') & (foo['Model'] == 'C/V'), 'RAM/C\/V/' + foo['Year'].astype(str),
    #                       foo['dir'])
    # foo['Path'] = np.where((foo['Make'] == 'RAM') & (foo['Model'] == 'C/V'),
    #                        foo['dir'] + '/' + foo['Path'].apply(lambda x: x.split('/')[-1]), foo['Path'])
    #
    # foo['count'] = foo.groupby(['Make', 'Model', 'Year'])['Path'].transform('count')
    # complete = foo.loc[foo['count'] >= opt.num_images][['Make', 'Model', 'Year']].drop_duplicates().reset_index(
    #     drop=True)


    # Remove make-model-year combinations where image count sufficient
    # df = df.merge(complete, on=['Make', 'Model', 'Year'], how='outer', indicator=True)
    # df = df.loc[df._merge != 'both'].reset_index(drop=True)
    # del df['_merge']

    # if opt.top:
    #     df = df.sort_values(by=['Make', 'Model', 'Year'], ascending=True)
    # else:
    #     df = df.sort_values(by=['Make', 'Model', 'Year'], ascending=False)


    path = "E:/TFM/Dataset1Car/images/train/"

    # Get a list of all the image files in the directory
    folder_files = [f for f in os.listdir(path)]

    wd = webdriver.Chrome(ChromeDriverManager().install())
    wd.get("https://google.com")

    for query in folder_files:
   # for i in range(len(folder_files)):
        #query = df.iloc[i, 0] + ' ' + df.iloc[i, 1] #+ ' ' + df.iloc[i, 3] + ' ' + str(df.iloc[i, 4])

        # # Ensuring directory structure right
        # if df.iloc[i, 2] == 'C/K':
        #     fix_model = 'C:K'
        # elif df.iloc[i, 2] == 'C/V':
        #     fix_model = 'C:V'
        # else:
        #     fix_model = df.iloc[i, 2]
        #
        # make_model_year = os.path.join(df.iloc[i, 0], fix_model, str(df.iloc[i, 4]))
        os.makedirs(os.path.join(opt.output_path, query.lower()), exist_ok=True)

        search_and_download(wd, query=query, root_dir_path=opt.output_path, make_model_year=query.lower(),
                            number_images=opt.num_images)


import os
import cv2

def arreglarFolder():

    two_names = ['alfa', 'romeo', 'aston', 'martin', 'land', 'rover']
    # Path to the directory containing the image files
    path = "E:/TFM/car/car_data/car_data/test/"
    path2 = "E:/TFM/Dataset1Car/images/test/"

    # Get a list of all the image files in the directory
    folder_files = [f for f in os.listdir(path)]

    # Create a new directory to store the renamed images
    new_folder_path = path2
    os.makedirs(new_folder_path, exist_ok=True)

    # Loop through each image file and rename it
    for name_file in folder_files:
        name_file = name_file.lower()
        image_files = [f for f in os.listdir(path + name_file) if f.endswith(".jpg") or f.endswith(".png")]
        for image_file in image_files:
            # Load the image using cv2
            image_file=image_file.lower()
            image_path = os.path.join(path + name_file, image_file)
            image = cv2.imread(image_path)

            if name_file.lower().split()[0] in two_names and name_file.lower().split()[1] in two_names:
                n = name_file.lower().split()
                name_file_new = n[0] + " " + n[1] + " " + n[2]
            else:
                n = name_file.lower().split()
                name_file_new = n[0] + " " + n[1]

            os.makedirs(os.path.join(path2, name_file_new), exist_ok=True)
            # Rename the image file using a new naming convention
            new_image_file = name_file_new + "/" + image_file

            # Save the renamed image to the new directory
            new_image_path = os.path.join(new_folder_path, new_image_file)
            cv2.imwrite(new_image_path, image)

    print("All images have been renamed and saved to the new directory.")


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)