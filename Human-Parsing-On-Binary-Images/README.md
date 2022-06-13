# Human Parsing On Binary Images


![4](https://user-images.githubusercontent.com/88143329/173433548-c6984acb-3348-46c3-ad4e-b2a162582983.png)


## Setup
Clone repo and install requirements.txt
  ```
  git clone https://github.com/ultralytics/yolov5  # clone
  cd Helmet-Detection
  pip install -r requirements.txt  # install
  ```
  
## Download & Extract Dataset
**About Dataset**:  [GitHub repo](https://github.com/Healthcare-Robotics/bodies-at-rest.git) 

Download and extract the dataset. the output images will be saved in ``./extracted_data`` directory.


  ```
  python data_extractor.py
  ```
  
## Dataset Preparation
- For best training you must Increase brightness
- Convert rgb images to segmentation images
- Use pascal dataset
- Pascal Person Part is a tiny single person human parsing dataset with 3000+ images. This dataset focus more on body parts segmentation. Pascal Person Part has 7 labels, including 'Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'.
- Prepared dataset will be saved in ``./Dataset`` directory


```
Dataset include:
|--- train_imgaes  # training binary person images with jpg format
|--- val_images # validation binary person images with jpg format
|--- train_segmentations  # training annotations grayscale person image with png format
|--- val_segmentations # validation annotations grayscale person image with png format
|--- train_id.txt # training image list
|--- val_id.txt # validation image list
```

```
python data_preparator.py --path [INPUT_PATH]
```
  
## Download Weights
  
  Download Weights will be saved in ``./weights`` directory and include:
  - [Pascal](https://drive.google.com/file/d/1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE/view?usp=sharing) 
  - [Resnet101](https://drive.google.com/file/d/19pVXyW6qxTHWC3-6gcU1kbQiesTBL9NA/view?usp=sharing) 
  - [Binary](https://drive.google.com/file/d/1sOCAg4anADBa1WGRBkDM_kMZooi7s69B/view?usp=sharing) 
  

   ```
   python get_weights.py
   ```
  
## Train
By default, the trained model will be saved in ``./log`` directory.

   ```
   python train.py --data-dir [INPUT_PATH] --num-classes [NUMBER-CLASSES] --batch-size [BATCH_SIZE] --epochs [INPUT_EPOCHS] --imagenet-pretrain [CHECKPOINT_PATH]
   ```

## Inference

   ```
   python simple_extractor.py --dataset [DATASET] --model-restore [CHECKPOINT_PATH] --input-dir [INPUT_PATH] --output-dir [OUTPUT_PATH]
   ```
