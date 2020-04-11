import os
import shutil

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# 指定された名称のディレクトリを作成する関数
def mkdir(d, rm=False):
    if rm:
        # 既存の同名ディレクトリがあれば削除
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    else:
        # 既存の同名ディレクトリがある場合は何もしない
        try: os.makedirs(d)
        except FileExistsError: pass


# 訓練用データセットを取得する関数
def make_generator(src_dir, valid_rate, input_size, batch_size):
    # ImageDataGenerator クラスのインスタンスを作成
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=valid_rate)

    # 訓練用データを読み込むためのジェネレータを作成
    train_generator = train_datagen.flow_from_directory(
        src_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    # 検証用データを読み込むためのジェネレータを作成
    valid_generator = train_datagen.flow_from_directory(
        src_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    return train_generator, valid_generator


# 訓練状況を可視化する関数
def plot(history, filename):
    # 訓練状況の折れ線グラフを描画する関数
    def add_subplot(nrows, ncols, index, xdata, train_ydata, valid_ydata, ylim, ylabel):
        plt.subplot(nrows, ncols, index)
        plt.plot(xdata, train_ydata, label='training', linestyle='--')
        plt.plot(xdata, valid_ydata, label='validation')
        plt.xlim(1, len(xdata))
        plt.ylim(*ylim)
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')

    # 描画領域のサイズを指定
    plt.figure(figsize=(10, 10))
    # x 軸のデータ(エポック数)を取得
    xdata = range(1, 1 + len(history['loss']))
    # 検証用データにおける損失を可視化
    add_subplot(2, 1, 1, xdata, history['loss'], history['val_loss'], (0, 5), 'loss')
    # 検証用データにおける正解率を可視化
    add_subplot(2, 1, 2, xdata, history['accuracy'], history['val_accuracy'], (0, 1), 'accuracy')
    # 可視化結果をファイルとして保存
    plt.savefig(filename)
    plt.close('all')