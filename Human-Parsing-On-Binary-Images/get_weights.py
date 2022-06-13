import os
import gdown

os.makedirs("weights",exist_ok=True)
urls = {
    "weights/imagenet.pth":"https://drive.google.com/uc?id=19pVXyW6qxTHWC3-6gcU1kbQiesTBL9NA",
    "weights/pascal.pth":"https://drive.google.com/uc?id=1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE",
    "weights/checkpoint_20.pth.tar":"https://drive.google.com/uc?id=1sOCAg4anADBa1WGRBkDM_kMZooi7s69B"
}
def download_weights():
    for out, url in urls.items():
        gdown.download(url,out)

if __name__ == "__main__":
    download_weights()
