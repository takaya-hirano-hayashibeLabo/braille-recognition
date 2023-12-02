from PIL import Image
import numpy as np

img_file = 'a.png'

# リサイズ前の画像を読み込み
img = Image.open(img_file)
# 読み込んだ画像の幅、高さを取得し半分に
scale=5
(width, height) = (img.width*scale, img.height*scale)
# 画像をリサイズする
img_resized = img.resize((width, height))
img_resized=np.array(img_resized)
img_resized[img_resized>80]=248
img_resized=Image.fromarray(img_resized,mode='L')
# ファイルを保存
img_resized.save('a_v2.png', quality=90)