   
import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import telebot

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help="Please Enter path of input model/  Example: model.h5")
arg = parser.parse_args()

model = load_model(arg.input_model)
bot = telebot.TeleBot("Token")

@bot.message_handler(commands=['start'])
def start(messages):
    bot.send_message(messages.chat.id, f'Hi {messages.from_user.first_name}')
    bot.send_message(messages.chat.id, "You can send a picture of rose, iris, dandelion and  tulip and the bot will recognize the type of flower for you.")

@bot.message_handler(content_types=['photo'])
def photo(message):
    file = bot.get_file(message.photo[-1].file_id)
    download = bot.download_file(file.file_path)
    path = file.file_path
    with open(path, 'wb') as f:
        f.write(download)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img/255
    img = img.reshape(1, 224,224, 3)
    pred = model.predict(img)

    res = np.argmax(pred)
    if res == 0:
      bot.reply_to(message, 'Desert')
    elif res == 1:
      bot.reply_to(message, 'Forest')
    elif res == 2:
      bot.reply_to(message, 'Mountain')
    elif res == 3:
      bot.reply_to(message, 'Sea')

bot.polling()