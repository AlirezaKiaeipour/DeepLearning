# X-Ray Pneumonia 

  **Dataset Link:** [Chest X-Ray](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
  
  **Models Link:** [Models](https://drive.google.com/drive/folders/1C44lOwXWsS9uHxB5YyoyQk4kfOoCz96N?usp=sharing)

  - Chest X-Ray Pneumonia using Transfer Learning

    ![2](https://user-images.githubusercontent.com/88143329/156312207-4279eafd-b6e1-4131-9288-a44fb2965304.png)

  - Preprocessing:
  
    - [x]  Our data is not balanced and I have to balance it
    
    ![3](https://user-images.githubusercontent.com/88143329/156312297-57caf4c2-192e-4429-ac78-46058d1ab06c.png)
    
    - [x] Balanced data using compute_class_weight from sklearn

  - Model:

    - [x]  MobileNetV2
    - [x]  InceptionV3
    - [x]  Xception


  - Accuracy:

    Algorithm | MobileNetV2 | InceptionV3 | Xception |
    ------------- | ------------- | ------------- | ------------- |
    Accuracy | **91.82 %** | **90.86 %** | **91.50 %**  |
    
    
