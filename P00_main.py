import os
import sys


# 訓練モード用の設定
TRN_SRC_DIR      = 'D00_dataset/training'
TRN_DST_DIR      = 'D01_estimator'
TRN_EST_FILE     = os.path.join(TRN_DST_DIR, 'estimator.h5')
TRN_CLS_FILE     = os.path.join(TRN_DST_DIR, 'class.pkl')
TRN_INFO_FILE    = os.path.join(TRN_DST_DIR, 'model_info.txt')
TRN_GRAPH_FILE   = os.path.join(TRN_DST_DIR, 'model_graph.pdf')
TRN_HIST_FILE    = os.path.join(TRN_DST_DIR, 'history.pdf')
TRN_FT_HIST_FILE = os.path.join(TRN_DST_DIR, 'ft_history.pdf')
TRN_INPUT_SIZE   = (160, 160)
TRN_DENSE_DIMS   = [4096, 2048, 1024, 128]
TRN_LR           = 1e-4
TRN_FT_LR        = 1e-5
TRN_MIN_LR       = 1e-7
TRN_MIN_FT_LR    = 1e-8
TRN_BATCH_SIZE   = 32
TRN_REUSE_CNT    = 10
TRN_EPOCHS       = 200
TRN_VALID_RATE   = 0.2
TRN_ES_PATIENCE  = 30
TRN_LR_PATIENCE  = 10
TRN_FT_START     = 15

# 推定モード用の設定
EST_SRC_DIR  = 'D00_dataset/test'
EST_DST_DIR  = 'D02_result'
EST_DRS_FILE = os.path.join(EST_DST_DIR, 'detailed_result.txt')
EST_SRS_FILE = os.path.join(EST_DST_DIR, 'summary_result.txt')


# 使用方法を表示
if len(sys.argv) == 1:
    print('Usage: python3 %s MODE...' % sys.argv[0])
    print('''
------------
Mode options
------------
trn: Makes an estimator.
est: Conducts estimation.
    ''')
    sys.exit(0)


# 訓練モードを実行
if 'trn' in sys.argv:
    print('Making an estimator...')

    n_class = len(os.listdir(TRN_SRC_DIR))
    TRN_DENSE_DIMS.append(n_class)

    from P01_model_maker import ModelMaker
    maker = ModelMaker(
        src_dir      = TRN_SRC_DIR,
        dst_dir      = TRN_DST_DIR,
        est_file     = TRN_EST_FILE,
        cls_file     = TRN_CLS_FILE,
        info_file    = TRN_INFO_FILE,
        graph_file   = TRN_GRAPH_FILE,
        hist_file    = TRN_HIST_FILE,
        ft_hist_file = TRN_FT_HIST_FILE,
        input_size   = TRN_INPUT_SIZE,
        dense_dims   = TRN_DENSE_DIMS,
        lr           = TRN_LR,
        ft_lr        = TRN_FT_LR,
        min_lr       = TRN_MIN_LR,
        min_ft_lr    = TRN_MIN_FT_LR,
        batch_size   = TRN_BATCH_SIZE,
        reuse_count  = TRN_REUSE_CNT,
        epochs       = TRN_EPOCHS,
        valid_rate   = TRN_VALID_RATE,
        es_patience  = TRN_ES_PATIENCE,
        lr_patience  = TRN_LR_PATIENCE,
        ft_start     = TRN_FT_START)
    maker.execute()

if 'est' in sys.argv:
    print('Conducting estimation...')

    from D02_estimator import Estimator
    estimator = Estimator(
        src_dir    = EST_SRC_DIR,
        dst_dir    = EST_DST_DIR,
        est_file   = TRN_EST_FILE,
        cls_file   = TRN_CLS_FILE,
        drs_file   = EST_DRS_FILE,
        srs_file   = EST_SRS_FILE,
        input_size = TRN_INPUT_SIZE)
    estimator.execute()
