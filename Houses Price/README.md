# Houses Price Estimation

  **Dataset Link:** [Houses Price](https://drive.google.com/drive/folders/1Rk415jadJbp5rkbfZQ0WZBikphv12--X?usp=sharing)

  - Houses Price Estimation using TensorFlow & Keras

  - Images included:
    
     - Bathroom
     - Bedroom
     - Frontal
     - Kitchen
    
  - Algorithm:

    - [x] Convolutional Neural Network(CNN)
    - [x] Multi Layer Perceptron(MLP)
    - [x] MultiChannel


  - Loss:

    Algorithm | MAE Loss |
    ------------- | ------------- |
    MultiChannel | **155890.0938** |
    

  - Inference:


      ## RUN

      You can run  Inference with the following command
      
      ```
      $ pip install requirements.txt
      
      python inference.py [--input_bedroom INPUT] [--input_bathroom INPUT] [--input_area INPUT] [--input_zipcode INPUT] [--input_image_path INPUT]
      ```
      
      
    ![2](https://user-images.githubusercontent.com/88143329/159670472-3098f776-eeac-4fcc-8317-c9579608b75b.png)


    
    Type | Real | Estimation |
    ------------- | ------------- | ------------- |
    Price | **889,000** | **840,210** |
