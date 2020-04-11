import os
import shutil
import sys

from keras.datasets import mnist
from keras.utils import to_categorical
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
def load_data(data_size=-1):
    (train_images, train_classes), (_, _) = mnist.load_data()

    train_images = train_images.reshape(60000, 28, 28, 1)
    train_images = train_images.astype('float32') / 255
    train_classes = to_categorical(train_classes)

    if data_size > len(train_images):
        # 指定されたデータ数がデータセットサイズを上回っていた場合のエラー処理
        print('[ERROR] data_size should be less than or equal to len(train_images).')
        sys.exit(0)
    elif data_size == -1:
        # データ数の指定が無い場合はデータセットサイズを取得するデータ数とする
        data_size = len(train_images)

    return train_images[:data_size], train_classes[:data_size]


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