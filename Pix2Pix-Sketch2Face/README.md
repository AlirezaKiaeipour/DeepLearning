# Sketch to Face

  **Dataset Link:** [Sketch](https://www.kaggle.com/datasets/alirezakiaipoor/sketch2face)
  
  **Model Link:** [Model](https://drive.google.com/drive/folders/17KUL7UMjOEiJK5NsLr1CDd4vLdODvP3k?usp=sharing)
  

  - Sketch to Face Pix2Pix using TensorFlow2.0 & Keras

    ![1](https://user-images.githubusercontent.com/88143329/163154565-b9644028-de65-4ff7-ab3c-e9ed6a01af6e.png)

    
  - Algorithm:

    - [x] Convolutional Neural Network(CNN)
    - [x] Image-to-image translation with a conditional GAN(Pix2Pix)
    

  - Inference:

       You can run  Inference with the following command

      ## RUN

      **Please Download [Model](https://drive.google.com/drive/folders/17KUL7UMjOEiJK5NsLr1CDd4vLdODvP3k?usp=sharing)**


      ```
      $ pip install requirements.txt
      
      python inference_image.py [--input_model INPUT] [--input_image INPUT]
      
      python inference_qt.py [--input_model INPUT]
      ```
      
