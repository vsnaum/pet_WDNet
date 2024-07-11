import telebot
from PIL import Image
from io import BytesIO
import WMRemoverNN as wmr


TOKEN = ''
bot = telebot.TeleBot(TOKEN, num_threads=1)

NOT_PHOTO = ["text", "audio", "document", "sticker", "video", "video_note", "voice", "location", "contact",
                 "new_chat_members", "left_chat_member", "new_chat_title", "new_chat_photo", "delete_chat_photo",
                 "group_chat_created", "supergroup_chat_created", "channel_chat_created", "migrate_to_chat_id",
                 "migrate_from_chat_id", "pinned_message"]


@bot.message_handler(commands=['start'])
def start(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, text="Hi! I'm watermark remover bot based on GAN. I was made by @vsnaumov as pet-project. Send me watermarked image and I'll try to reconstruct the original image.")

@bot.message_handler(content_types=['photo'])
def proc_photo(message):
    chat_id = message.chat.id
    img = bot.get_file(message.photo[-1].file_id) # max_size 1280x1280
    img = bot.download_file(img.file_path) # open img as bytes
    img = Image.open(BytesIO(img)) # read bytes as PIL

    max_size = 512
    if max(img.size) > max_size:
        aspect = img.height / img.width
        if aspect > 1:
            new_h = max_size
            new_w = int(new_h / aspect)
        elif aspect < 1:
            new_w = max_size
            new_h = int(aspect * new_w)
        elif aspect == 1:
            new_h, new_w = max_size,max_size
        bot.send_message(chat_id,f"Your image is too big for my limited pet-project resources. Due to memory limits I'm able to process images only up to {max_size}x{max_size} px. So I *reduce resolution to {new_w}x{new_h} px* before removing the watermark. Sorry 🥺",parse_mode= 'Markdown')
        img = img.resize((new_w,new_h))
    
    bot.send_message(chat_id,"Please, wait. I'm processing your image ⏳")
    img_nowm = wmr.run('WDnet_G.onnx',img)
    buffer = BytesIO()
    img_nowm.save(buffer,'JPEG')
    buffer.seek(0)
    bot.send_photo(chat_id,photo=buffer,reply_to_message_id=message.id)

@bot.message_handler(content_types=NOT_PHOTO)
def default_proc(message):
    chat_id = message.chat.id
    bot.send_message(chat_id,'Send me image as image (not as document)',reply_to_message_id=message.id)


print('start polling')
bot.polling(none_stop=True, interval=0)