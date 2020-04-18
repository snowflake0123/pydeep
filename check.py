from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


IMG_FILE = 'D00_dataset/training/scissors/scissors_0003.jpg'
PLT_ROW  = 1
PLT_COL  = 4


# 指定された datagen で画像の読み込み・加工処理を行い，結果を表示する関数
def plot(title, img, datagen):
    plt.figure(title)
    i = 0
    for data in datagen.flow(img, batch_size=1):
        plt.subplot(PLT_ROW, PLT_COL, i + 1)
        plt.axis('off')
        plt.imshow(array_to_img(data[0]))
        i += 1
        if i == PLT_ROW * PLT_COL:
            break
    plt.show()


# 画像を読み込む
img = load_img(IMG_FILE, target_size=(160, 160))
# 画像を(160, 160, 3)の形状の NumPy 配列に変換
img = img_to_array(img)
# NumPy 配列を(1, 160, 160, 3)の形状に変換
img = img.reshape((1,) + img.shape)

# 回転
datagen = ImageDataGenerator(rotation_range=30)
plot('rotation', img, datagen)

# 水平方向の平行移動
datagen = ImageDataGenerator(width_shift_range=0.2)
plot('width_shift', img, datagen)

# 垂直方向の平行移動
datagen = ImageDataGenerator(height_shift_range=0.2)
plot('height_shift', img, datagen)

# 歪み
datagen = ImageDataGenerator(shear_range=30)
plot('shear', img, datagen)

# ズーム
datagen = ImageDataGenerator(zoom_range=[0.7, 1.3])
plot('zoom', img, datagen)

# 水平方向の反転
datagen = ImageDataGenerator(horizontal_flip=True)
plot('horizontal_flip', img, datagen)

# 垂直方向の反転
datagen = ImageDataGenerator(vertical_flip=True)
plot('vertical_flip', img, datagen)

# 上記全てのデータ加工
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=30,
    zoom_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True)
plot('all', img, datagen)