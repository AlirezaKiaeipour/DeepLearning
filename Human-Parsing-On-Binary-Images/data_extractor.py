import os
import glob
import pickle
import requests
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

urls = ["https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/1X6YXX",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/ISZ8YM",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/AFZRHB",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/ZULONP",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/MHOB7T",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/VBQDNS",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/JGHRAW",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/EGBFOR",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/9570H2",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/Z6JVFZ",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/ZJJWKC",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/WWIR58",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/KZDPBA",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/Y1W6TB",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/ZH75SS",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/CEQYPP",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/WOPOXU",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/UMW2MI",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/7EPFLF",
        "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/KOA4ML/XON1LX"]

# Download Dataset
def download_data():
    for index, url in enumerate(tqdm(urls)):
        os.makedirs(f"./Download/{index}",exist_ok=True)
        response = requests.get(url)
        open(f"./Download/{index}/prescribed.p", "wb").write(response.content)

# Extract Dataset
def extract_data():
    for index,path in enumerate(os.listdir("./Download")):
        path = glob.glob(os.path.join(f"./Download/{path}/*"))
        for file in path:
            with open(file,"rb") as f:
                data = pickle.load(f,encoding="latin")
                for image in range(len(data["RGB"])):
                    img = np.array(data["RGB"][image])
                    img = Image.fromarray(img[...,::-1])
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.5) #  Increase brightness
                    os.makedirs("extracted_data",exist_ok=True)
                    img.save(f"extracted_data/image_{index}_{image}.jpg")

if __name__ == "__main__":
    download_data()
    extract_data()
    