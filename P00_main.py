import os


DST_DIR     = 'D01_estimator'
EST_FILE    = os.path.join(DST_DIR, 'estimator.h5')
INFO_FILE   = os.path.join(DST_DIR, 'model_info.txt')
GRAPH_FILE  = os.path.join(DST_DIR, 'model_graph.png')
HIST_FILE   = os.path.join(DST_DIR, 'history.pdf')
INPUT_SIZE  = (28, 28)
FILTERS     = (32, 64)
KERNEL_SIZE = (3, 3)
POOL_SIZE   = (2, 2)
DENSE_DIMS  = (1024, 128, 10)
LR          = 1e-3
DATA_SIZE   = 1000
BATCH_SIZE  = 128
EPOCHS      = 30
VALID_RATE  = 0.2


from P01_model_maker import ModelMaker
maker = ModelMaker (
    dst_dir     = DST_DIR,
    est_file    = EST_FILE,
    info_file   = INFO_FILE,
    graph_file  = GRAPH_FILE,
    hist_file   = HIST_FILE,
    input_size  = INPUT_SIZE,
    filters     = FILTERS,
    kernel_size = KERNEL_SIZE,
    pool_size   = POOL_SIZE,
    dense_dims  = DENSE_DIMS,
    lr          = LR,
    data_size   = DATA_SIZE,
    batch_size  = BATCH_SIZE,
    epochs      = EPOCHS,
    valid_rate  = VALID_RATE
)
maker.execute()