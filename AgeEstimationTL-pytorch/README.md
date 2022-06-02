# Age Estimation


  **Dataset Link:** [Age-utkface](https://www.kaggle.com/jangedoo/utkface-new)
  
  **Model Link:** [Model](https://drive.google.com/file/d/1MTcZQVhlZK3Vmy3VMNdD4i0MrdRxmlFA/view?usp=sharing)

  - Age Estimation using pytorch

  - Hyperparameter Tuning using W&B Sweep

  - Model:

    - [x]  Resnet152


  - Loss:

    Algorithm | MSE-Loss |
    ------------- | ------------- |
    Resnet152 | **25.86** |
    

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
      python test.py [--input_path INPUT] [--input_weights INPUT] [--input_device INPUT]
      ```  
      
    ## Inference
           
      ```
      python inference.py [--input_weights INPUT] [--input_device INPUT] [--input_image INPUT]
      ```  
