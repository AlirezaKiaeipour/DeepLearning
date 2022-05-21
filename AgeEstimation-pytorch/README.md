# Age Estimation


  **Dataset Link:** [Age-utkface](https://www.kaggle.com/jangedoo/utkface-new)
  
  **Model Link:** [Model](https://drive.google.com/file/d/1joypkcaMGNRUwpjsmqHW5IlZdq5aBq4e/view?usp=sharing)

  - Age Estimation using pytorch

  - Model:

    - [x]  Convolutional Neural Network(CNN)


  - Accuracy & Loss:

    Algorithm | Loss |
    ------------- | ------------- |
    Personal Model | **383.42** |
    

      ## Installation
      
       **Please Clone Repository**
       
      ```
      $ pip install requirements.txt
      ```
      

     ## Train
           
      ```
      python train.py [--input_path INPUT] [--input_device INPUT] [--input_epochs INPUT]
      ```                             

    ## Test
           
      ```
      python test.py [--input_path INPUT] [--input_device INPUT]
      ```  
      
    ## Inference
           
      ```
      python inference.py [--input_device INPUT] [--input_image INPUT]
      ```  
