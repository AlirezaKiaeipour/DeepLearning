# Emotion Detection

  **Dataset Link:** [Emotion](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)
  
  **Model Link:** [Model](https://drive.google.com/drive/folders/12PZHQGxawpbIkwH4TZz1Z6aF9UXi20pf?usp=sharing)

  - Emotion Detection using TensorFlow and Keras
    
  - Model:

    - [x] Convolutional Neural Network(CNN)


  - Accuracy & Loss:

    Algorithm | Accuracy | Loss |
    ------------- | ------------ | ------------- |
    Personal Model | **68.57 %** | **0.9192** |
    

  - Inference:

      ## RUN
      You can run  Inference with the following command
      
      **Please Download [Model](https://drive.google.com/drive/folders/12PZHQGxawpbIkwH4TZz1Z6aF9UXi20pf?usp=sharing)**

      ```
      $ pip install requirements.txt
      
      python inference_image.py [--input_model INPUT] [--input_image INPUT]
      
      python inference_webcam.py [--input_model INPUT]
      
      python inference_qt.py [--input_model INPUT]
      ```

      ![14](https://user-images.githubusercontent.com/88143329/165132167-f7ee0eaa-0435-4e07-91bd-9a9c616fe758.png)
      ![12](https://user-images.githubusercontent.com/88143329/165132261-55a244d5-8ada-4ea9-b21a-556d7058f22b.png)
      ![15](https://user-images.githubusercontent.com/88143329/165132294-6a4d7924-753b-45a2-a95e-3099f1c52ab6.png)

