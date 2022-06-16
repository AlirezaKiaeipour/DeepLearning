from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

class Mymodel(Model):
  def __init__(self):
      super().__init__()
      self.conv2D_1 = Conv2D(64,(3,3),activation="relu",input_shape=(224,224,3))
      self.conv2D_2 = Conv2D(128,(3,3),activation="relu")
      self.maxpool2D = MaxPool2D()
      self.dense1 = Dense(256,activation="relu")
      self.dense2 = Dense(14,activation="softmax")
      self.flatten = Flatten()
      self.dropout = Dropout(0.5)
    
  def call(self,x):
    x = self.conv2D_1(x)
    x = self.maxpool2D(x)
    x = self.conv2D_2(x)
    x = self.maxpool2D(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dropout(x)
    output = self.dense2(x)

    return output
