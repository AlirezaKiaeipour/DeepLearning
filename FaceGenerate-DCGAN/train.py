import os
import argparse
import time
import glob
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--input_epoch",type=int,help="Please Enter number of epoch/  Example: 200")
arg = parser.parse_args()

images = []
for image in glob.glob(os.path.join("Dataset/Train/Male/*"))[0:50000]:
    image = cv2.imread(image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(64,64))
    image = (image - 127.5) / 127.5
    images.append(image)
images = np.array(images).astype("float32")
train_data = tf.data.Dataset.from_tensor_slices(images).shuffle(2000).batch(256)

model = Model()
gen_model = model.generator()
disc_model = model.discriminator()

class Train:
    def __init__(self):
        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.gen_optimizer = tf.keras.optimizers.Adam()
        self.disc_optimizer = tf.keras.optimizers.Adam()
        self.disc_loss = tf.keras.metrics.Mean()
        self.seed = tf.random.normal([16,100])

    def train_step(self,images):
        noise = tf.random.normal([256,100])
        with tf.GradientTape() as gTape_gen, tf.GradientTape() as gTape_disc:
            generated_images = gen_model(noise,training=True)
            real_images = disc_model(images,training=True)
            fake_images = disc_model(generated_images,training=True)

            loss_gen = self.loss_function(tf.ones_like(fake_images),fake_images)
            loss_real = self.loss_function(tf.ones_like(real_images),real_images)
            loss_fake = self.loss_function(tf.zeros_like(fake_images),fake_images)
            loss_disc = loss_real + loss_fake
            self.disc_loss(loss_disc)
        
        gradient_gen = gTape_gen.gradient(loss_gen,gen_model.trainable_variables)
        gradient_disc = gTape_disc.gradient(loss_disc,disc_model.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(gradient_gen,gen_model.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradient_disc,disc_model.trainable_variables))

    def save_images(self,gen_model,epoch,seed):
        prediction = gen_model(seed, training=False)
        fig = plt.figure(figsize=(4, 4))
        for i in range(prediction.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow((np.array(prediction[i, :, :, :] * 127.5)+127.5).astype('uint8'))
            plt.axis('off')
        plt.savefig(f"output/image{epoch}.png")
        plt.clf()

    def gif(self):
        with imageio.get_writer("Face_Generated.gif", mode='I') as writer:
            filenames = glob.glob('output/image*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

    def train(self):
        for epoch in range(arg.input_epoch):
            start = time.time()
            self.disc_loss.reset_states()  
            for images in tqdm(train_data):
                self.train_step(images)
            self.save_images(gen_model,epoch + 1,self.seed)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            print ('Loss: {} '.format(self.disc_loss.result()))

if __name__ == "__main__":
    image_generator = Train()
    image_generator.train()
    image_generator.gif()
    