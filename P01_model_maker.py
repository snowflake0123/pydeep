from keras.layers import Flatten, Input
from keras.models import Model
from keras.optimizers import Adam

import P10_util as util
import P11_model_util as mutil


class ModelMaker:
    # コンストラクタ
    def __init__(self, dst_dir, est_file,
                 info_file, graph_file, hist_file,
                 input_size, filters, kernel_size, pool_size, dense_dims,
                 lr, data_size, batch_size, epochs, valid_rate):
        self.dst_dir     = dst_dir
        self.est_file    = est_file
        self.info_file   = info_file
        self.graph_file  = graph_file
        self.hist_file   = hist_file
        self.input_size  = input_size
        self.filters     = filters
        self.kernel_size = kernel_size
        self.pool_size   = pool_size
        self.dense_dims  = dense_dims
        self.lr          = lr
        self.data_size   = data_size
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.valid_rate  = valid_rate


    # モデルを定義するメソッド
    def define_model(self):
        # 入力層の定義
        input_x = Input(shape=(*self.input_size, 1))
        x = input_x

        # 畳み込み層・プーリング層の定義
        for f in self.filters:
            x = mutil.add_conv_pool_layers(
                x, f, self.kernel_size, self.pool_size)

        # 平滑化層の定義
        x = Flatten()(x)

        # 全結合層の定義
        for dim in self.dense_dims[:-1]:
            x = mutil.add_dense_layer(x, dim)

        # 出力層の定義
        x = mutil.add_dense_layer(
            x, self.dense_dims[-1], activation='softmax')

        # モデル全体の入出力を定義
        model = Model(input_x, x)

        # モデルの訓練条件の設定
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        return model


    # モデルを訓練するメソッド
    def fit_model(self, model):
        # データセットの読み込み
        train_images, train_classes = util.load_data(self.data_size)

        # 訓練の実行
        history = model.fit(
            train_images,
            train_classes,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.valid_rate)

        return model, history.history


    # プログラム全体を制御するメソッド
    def execute(self):
        # モデルを定義
        model = self.define_model()
        # モデルを訓練
        model, history = self.fit_model(model)

        # 訓練したモデルを保存
        util.mkdir(self.dst_dir, rm=True)
        model.save(self.est_file)

        # ネットワーク構造を保存
        mutil.save_model_info(self.info_file, self.graph_file, model)

        # 訓練状況を保存
        if 'acc' in history:
            history['accuracy'] = history.pop('acc')
            history['val_accuracy'] = history.pop('val_acc')
        util.plot(history, self.hist_file)

        # 最終エポックの検証用データにおける損失を標準出力
        print('val_loss: %f' % history['val_loss'][-1])