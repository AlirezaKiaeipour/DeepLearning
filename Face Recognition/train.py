import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from model import Mymodel

parser = argparse.ArgumentParser()
parser.add_argument("--input_dataset",type=str,help="Please Enter path of input dataset")
arg = parser.parse_args()

image_generator = ImageDataGenerator(
    rescale = 1.0 / 255.0,
    horizontal_flip = True,
    validation_split = 0.2
)

path = arg.input_dataset
Train_images = image_generator.flow_from_directory(
    path,
    class_mode = "categorical",
    batch_size = 32,
    target_size = (224,224),
    subset = "training"
)
Val_images = image_generator.flow_from_directory(
    path,
    class_mode = "categorical",
    batch_size = 32,
    target_size = (224,224),
    subset = "validation"
)

model = Mymodel()

class Fit:
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()
        self.accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.loss = tf.keras.metrics.MeanAbsoluteError()
        self.val_loss = tf.keras.metrics.MeanAbsoluteError()
        self.epochs = 20
    
    def train_step(self,images,labels):
        with tf.GradientTape() as gTape:
            logits = model(images)
            loss_value = self.loss_function(labels,logits)
            self.accuracy(labels,logits)
            self.loss(labels,logits)

        gradients = gTape.gradient(loss_value,model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    def test_step(self,images,labels):
        logits = model(images)
        loss_value = self.loss_function(labels,logits)
        self.val_accuracy(labels,logits)
        self.val_loss(labels,logits)
    
    def train(self):
        for epoch in range(self.epochs):
            self.accuracy.reset_states()
            self.val_accuracy.reset_states()
            self.loss.reset_states()
            self.val_loss.reset_states()
            print("epochs:",epoch)

            for index,(images,labels) in enumerate(tqdm(Train_images)):
                self.train_step(images,labels)
                if len(Train_images) <= index:
                    break

            for index,(images,labels) in enumerate(tqdm(Val_images)):
                self.test_step(images,labels)
                if len(Val_images) <= index:
                    break
            
            print("loss:",self.loss.result())
            print("accuracy:",self.accuracy.result())
            print("val_loss",self.val_loss.result())
            print("val_accuracy",self.val_accuracy.result())

    def save_model(self):
        model.save_weights(filepath='weights/face_recognition')

if __name__ == "__main__":
    Face_Recognition = Fit()
    Face_Recognition.train()
    Face_Recognition.save_model()
    