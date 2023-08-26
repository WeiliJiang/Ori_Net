class hparams:
    train_or_test = 'train'
    class_ = 10
    ori_class=146
    debug = False
    mode = '3d'
    in_class = 1
    out_class = 1
    crop_or_pad_size = 512
    patch_size = 128
    fold_arch = '*.nii.gz'
    source_train_dir = r'G:\CCTA_data\COR_oriention\train\image'
    vessel_train_dir = r'G:\CCTA_data\COR_oriention\train\label\vessel'
    oriention_train_dir=r'G:\CCTA_data\COR_oriention\train\label\oriention'

    source_test_dir = r'G:\CCTA_data\COR_oriention\test\image'
    vessel_test_dir = r'G:\CCTA_data\COR_oriention\test\label\vessel'
    oriention_test_dir =r'G:\CCTA_data\COR_oriention\test\label\oriention'

    output_dir_test = r'H:\oriention_seg\results\ori_predict64'