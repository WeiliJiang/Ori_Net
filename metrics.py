import torchio as tio
from pathlib import Path
import torch
import numpy as np
import copy
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)


predict_dir = r'H:\experiments\pla\results\GuideNet\prevessel'
# predict_dir = r'H:\oriention_seg\results\CAD_ori\ori_refine'
labels_dir = r'H:\experiments\pla\results\GuideNet\vessel'

# predict_dir = '/data0/my_project/med/seg_3d/results'
# labels_dir = '/data2/zkndataset/med/unet/label'


def do_subject(image_paths, label_paths):
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            pred=tio.ScalarImage(image_path),
            gt=tio.LabelMap(label_path),
        )
        subjects.append(subject)

images_dir = Path(predict_dir)
labels_dir = Path(labels_dir)

image_paths = sorted(images_dir.glob('*.nii.gz'))
label_paths = sorted(labels_dir.glob('*.nii.gz'))


subjects = []
do_subject(image_paths, label_paths)

training_set = tio.SubjectsDataset(subjects)


toc = ToCanonical()

dice_value =[]
jaccard = []
fpr = []
fnr=[]
precision=[]
recall=[]
for i,subj in enumerate(training_set):
    gt = subj['gt'][tio.DATA]


    # subj = toc(subj)
    pred = subj['pred'][tio.DATA]#.permute(0,1,3,2)

    # preds.append(pred)
    # gts.append(gt)

    preds = pred.numpy()
    gts = gt.numpy()



    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or
    fp_array = copy.deepcopy(pred)  # keep pred unchanged
    fn_array = copy.deepcopy(gdth)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision1 = tp / (pred_sum + smooth)
    recall1 = tp / (gdth_sum + smooth)

    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    jaccard1 = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)
    print('name', subj['gt']['stem'],'dice',dice)
    dice_value.append(dice)
    jaccard.append(jaccard1)
    fpr.append(false_positive_rate)
    fnr.append(false_negtive_rate)
    precision.append(precision1)
    recall.append(recall1)
# dice_value.remove(np.min(dice_value))
# dice_value.remove(np.min(dice_value))
#
# jaccard.remove(np.min(jaccard))
# jaccard.remove(np.min(jaccard))
#
# precision.remove(np.min(precision))
# precision.remove(np.min(precision))
#
# recall.remove(np.min(recall))
# recall.remove(np.min(recall))
# print('jaccard', np.mean(jaccard), np.std(jaccard))
# print('dice', np.mean(dice_value), np.std(dice_value))
# print('precision', np.mean(precision), np.std(precision))
# print('recall', np.mean(recall), np.std(recall))
# # print('*' * 10)
# print('jaccard', jaccard)
# print('dice_value', dice_value)
# print('precision', precision)
# print('recall', recall)