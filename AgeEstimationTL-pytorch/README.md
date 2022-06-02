# Persian Mnist


  **Dataset Link:** [PersianMnist](https://drive.google.com/drive/folders/1NNgHrH5bu8ib4z5xciDHdhEP4fmvQv2o?usp=sharing)
  
  **Model Link:** [Model](https://drive.google.com/file/d/15zd5_zM_xmqlw3EocyRPNQNpZ4vUM38q/view?usp=sharing)

  - Persian Mnist using pytorch
  
  - Hyperparameter Tuning using W&B Sweep

  - Model:

    - [x]  Resnet152


  - Accuracy & Loss:

    Algorithm | Accuracy | Loss |
    ------------- | ------------- | ------------- |
    Personal Model | **100.00 %** | **0.00117** |
    

      ## Installation
      
       **Please Clone Repository**
       
      ```
      $ pip install requirements.txt
      ```
      

     ## Train
           
      ```
      python train.py [--input_device INPUT] [--input_epochs INPUT]
      ```                             

    ## Test
           
      ```
      python test.py [--input_weights INPUT] [--input_device INPUT]
      ```  
      
    ## Inference
           
      ```
      python inference.py [--input_weights INPUT] [--input_device INPUT] [--input_image INPUT]
      ```  
