import math

from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers import Adam

import P10_util as util
import P11_model_util as mutil


class ModelMaker:

    # コンストラクタ
    def __init__(self, src_dir, dst_dir, est_file,
                 info_file, graph_file, hist_file, ft_hist_file,
                 input_size, dense_dims, lr, ft_lr, min_lr, min_ft_lr,
                 batch_size, reuse_count, epochs, valid_rate,
                 es_patience, lr_patience, ft_start):
        self.src_dir      = src_dir
        self.dst_dir      = dst_dir
        self.est_file     = est_file
        self.info_file    = info_file
        self.graph_file   = graph_file
        self.hist_file    = hist_file
        self.ft_hist_file = ft_hist_file
        self.input_size   = input_size
        self.dense_dims   = dense_dims
        self.lr           = lr
        self.ft_lr        = ft_lr
        self.min_lr       = min_lr
        self.min_ft_lr    = min_ft_lr
        self.batch_size   = batch_size
        self.reuse_count  = reuse_count
        self.epochs       = epochs
        self.valid_rate   = valid_rate
        self.es_patience  = es_patience
        self.lr_patience  = lr_patience
        self.ft_start     = ft_start


    # モデルを定義するメソッド
    def define_model(self):
        # VGG16 の平滑化層の手前までのベース部分を取得
        base_model = VGG16(include_top=False, input_shape=(*self.input_size, 3))

        # ベース部分の全層を凍結
        for layer in base_model.layers:
            layer.trainable = False

        # ベース部分の最終層を x として保持
        x = base_model.output

        # 平滑化層の定義
        x = Flatten()(x)

        # 全結合層の定義
        for dim in self.dense_dims[:-1]:
            x = mutil.add_dense_layer(x, dim)

        # 出力層の定義
        x = mutil.add_dense_layer(
            x, self.dense_dims[-1], use_bn=False, activation='softmax')

        # モデル全体の入出力を定義
        model = Model(base_model.input, x)

        # モデルの訓練条件の設定
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        return model


    # 特定層以降の凍結を解除するメソッド
    def unfreeze_layers(self, model):
        # self.ft_start 以降の層の凍結を解除
        for layer in model.layers[self.ft_start:]:
            layer.trainable = True

        # モデルの訓練条件を再設定
        model.compile(
            optimizer=Adam(lr=self.ft_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        return model


    # モデルを訓練するメソッド
    def fit_model(self, model):
        # (1) データセット読み込みのためのジェネレータを取得
        train_generator, valid_generator = util.make_generator(
            self.src_dir, self.valid_rate, self.input_size, self.batch_size)

        # (2) 1回目訓練用のコールバックを定義
        early_stopping = EarlyStopping(
            patience=self.es_patience,
            restore_best_weights=True,
            verbose=1)
        reduce_lr_op = ReduceLROnPlateau(
            patience=self.lr_patience,
            min_lr=self.min_lr,
            verbose=1)
        callbacks = [early_stopping, reduce_lr_op]

        # (3) 1回目訓練を実行
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=math.ceil(
                train_generator.n/self.batch_size)*self.reuse_count,
            epochs=self.epochs,
            validation_data=valid_generator,
            validation_steps=math.ceil(
                valid_generator.n/self.batch_size)*self.reuse_count,
            callbacks=callbacks)

        # (4) 1回目訓練済みのモデルの特定層以降の凍結を解除
        model = self.unfreeze_layers(model)

        # (5) ファインチューニング用のコールバックを定義
        reduce_lr_op = ReduceLROnPlateau(
            patience=self.lr_patience,
            min_lr=self.min_ft_lr,
            verbose=1)
        callbacks = [early_stopping, reduce_lr_op]

        # (6) ファインチューニングを実行
        ft_history = model.fit_generator(
            train_generator,
            steps_per_epoch=math.ceil(
                train_generator.n/self.batch_size)*self.reuse_count,
            epochs=self.epochs,
            validation_data=valid_generator,
            validation_steps=math.ceil(
                valid_generator.n/self.batch_size)*self.reuse_count,
            callbacks=callbacks)

        # (7) モデル，1回目訓練・ファインチューニングの訓練状況を返却
        return model, history.history, ft_history.history


    # プログラム全体を制御するメソッド
    def execute(self):
        # モデルを定義
        model = self.define_model()
        # モデルを訓練
        model, history, ft_history = self.fit_model(model)

        # 訓練したモデルを保存
        util.mkdir(self.dst_dir, rm=True)
        model.save(self.est_file)

        # ネットワーク構造を保存
        mutil.save_model_info(self.info_file, self.graph_file, model)

        # 訓練状況を保存
        if 'acc' in history:
            history['accuracy'] = history.pop('acc')
            history['val_accuracy'] = history.pop('val_acc')
            ft_history['accuracy'] = ft_history.pop('acc')
            ft_history['val_accuracy'] = ft_history.pop('val_acc')
        util.plot(history, self.hist_file)
        util.plot(ft_history, self.ft_hist_file)

        def get_min(loss):
            min_val = min(loss)
            min_ind = loss.index(min_val)
            return min_val, min_ind

        # 検証用データにおける最小の損失を標準出力（1回目訓練終了後）
        print('Before fine-tuning')
        min_val, min_ind = get_min(history['val_loss'])
        print('val_loss: %f (Epoch: %d)' % (min_val, min_ind + 1))

        # 検証用データにおける最小の損失を標準出力（ファインチューニング後）
        print('After fine-tuning')
        min_val, min_ind = get_min(ft_history['val_loss'])
        print('val_loss: %f (Epoch: %d)' % (min_val, min_ind + 1))