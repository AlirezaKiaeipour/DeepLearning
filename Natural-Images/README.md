# Natural

  **Dataset Link:** [Natural](https://drive.google.com/drive/folders/1I7C-KGVJPECn6S4Agz8a3dunI0YMln7t?usp=sharing)
  
  **Model Link:** [Model](https://drive.google.com/file/d/1w7UiW1jkUn7klOmQZQnQj_FHt93XtIu3/view?usp=sharing)
  

  - Natural image detection

    ![5](https://user-images.githubusercontent.com/88143329/158018306-adfdde94-5193-4c3c-ad97-6c1c1069f292.png)
 
  - Algorithm:
    - [x]  Convolutional Neural Network(CNN)

  - Accuracy & Loss:
    
      Data  | Accuracy | Loss |
    ------------- | ------------- | ------------- |
    Train | **72.37 %** | **0.7087** |
    Validation | **76.61 %** | **0.6481** |
    Test | **78.01 %** | **0.6122** |
    
    
      ## RUN
      You can run  Inference with the following command
      
      **Please download the [Model](https://drive.google.com/file/d/1w7UiW1jkUn7klOmQZQnQj_FHt93XtIu3/view?usp=sharing) first**

      ```
      python inference.py [--input_model INPUT] [--input_image INPUT]
      
      python Natural_bot.py [--input_model INPUT]
      ```
      
