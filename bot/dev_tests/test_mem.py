from PIL import Image
from io import BytesIO
import WMRemoverNN as wmr
import math
import tracemalloc

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


def app():
    img=Image.open('wm_huge2.jpg')

    max_size = 512
    max_vol = max_size*max_size
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
        img = img.resize((new_w,new_h))

    img_nowm = wmr.run('WDnet_G.onnx',img)
    buffer = BytesIO()
    img_nowm.save(buffer,'JPEG')
    buffer.seek(0)



tracemalloc.start()
app()
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()

print(f'peak memory: {convert_size(peak)}')