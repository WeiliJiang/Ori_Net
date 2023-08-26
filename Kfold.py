import os
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import glob
image_path='/root/workspace/DeepcadData/data_preprocessing/COR_ROI/image'
label_path='/root/workspace/DeepcadData/data_preprocessing/COR_ROI/label'
images1 = sorted(glob.glob(os.path.join(image_path, '*.nii.gz')))
labels1 = sorted(glob.glob(os.path.join(label_path, '*.nii.gz')))
all_files = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(images1, labels1)]
# print(len(all_files))
floder = KFold(n_splits=5,shuffle=False)
train_files = []   # 存放5折的训练集划分
test_files = []     # # 存放5折的测试集集划分
for k, (Trindex, Tsindex) in enumerate(floder.split(all_files)):
    print('Trindex:',Trindex)
    train_files.append(np.array(all_files)[Trindex].tolist())
    test_files.append(np.array(all_files)[Tsindex].tolist())

# 把划分写入csv,检验每次是否相同
df = pd.DataFrame(data=train_files, index=['0', '1', '2', '3', '4'])
df.to_csv('/root/workspace/DeepcadData/data_preprocessing/COR_ROI/kfold/train_patch.csv')
df1 = pd.DataFrame(data=test_files, index=['0', '1', '2', '3', '4'])
df1.to_csv('/root/workspace/DeepcadData/data_preprocessing/COR_ROI/kfold/test_patch.csv')
train_image_path = []
train_label_path = []
for i in range(len(train_files[0])):
    train_image_path.append(train_files[0][i]['image'])
    train_label_path.append(train_files[0][i]['label'])
print('train_file_path:',len(train_image_path))
# print('test_file_path:',len(train_label_path))
test_image_path = []
test_label_path = []
for i in range(len(test_files[0])):
    test_image_path.append(test_files[0][i]['image'])
    test_label_path.append(test_files[0][i]['label'])
print('test_file_path:',len(test_image_path))
# print('test_file_path:',test_label_path)