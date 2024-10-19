import glob
import os.path

from PIL import Image, ImageFilter, ImageOps

basepath = r'c:\prj\fns-captcha\pics'
files = glob.glob(r'c:\prj\fns-captcha\pics\*.jpg')

for f in files:
    with Image.open(f) as img:
        img.load()
    filename = os.path.basename(f).replace(".jpg", '.png')
    file_save_path = os.path.join(basepath, 'png', filename)
    img_gray = img.convert("L")
    edges = img_gray.filter(ImageFilter.FIND_EDGES)
    inv_img = ImageOps.invert(edges)
    # inv_img = inv_img.reduce(2)
    # inv_img.show()
    inv_img.save(file_save_path)

