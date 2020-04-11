from keras.layers import Conv2D, Dense, MaxPooling2D
from keras.utils import plot_model


# 畳み込み層とプーリング層を追加する関数
def add_conv_pool_layers(x, filters, kernel_size, pool_size, activation='relu'):
    x = Conv2D(filters, kernel_size, padding='same', activation=activation)(x)
    x = Conv2D(filters, kernel_size, padding='same', activation=activation)(x)
    x = MaxPooling2D(pool_size)(x)
    return x


# 全結合層を追加する関数
def add_dense_layer(x, dim, activation='relu'):
    x = Dense(dim, activation=activation)(x)
    return x


# ネットワーク構造を可視化して保存する関数
def save_model_info(info_file, graph_file, model):
    with open(info_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    plot_model(model, to_file=graph_file, show_shapes=True)