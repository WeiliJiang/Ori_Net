from skimage.morphology import skeletonize_3d
import nibabel as nb
import numpy as np
from nilearn.image import resample_img
from skimage.measure import label
import math
import os

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    new_arr=np.zeros_like(data)
    # print('1',new_arr.shape[2])
    for i in range(new_arr.shape[2]):
        # nonzero_mask = np.zeros_like(data[:,:,i])
        this_mask = data[:,:,i]
        # nonzero_mask = nonzero_mask | this_mask
        nonzero_mask = binary_fill_holes(this_mask)
        new_arr[:,:,i]=nonzero_mask
    return new_arr

img_path=r'G:\CCTA_data\COR_oriention\test\image'
label_path=r'G:\CCTA_data\COR_oriention\test\label\vessel'

img_save=r'G:\CCTA_data\COR_oriention\test\image'
vessel_save=r'G:\CCTA_data\COR_oriention\test\label\vessel'
oriention_save=r'G:\CCTA_data\COR_oriention\test\label\ceshi'

img_id=os.listdir(img_path)
for idi in img_id:
    img_file=os.path.join(img_path,idi)
    label_file =os.path.join(label_path,idi)
    img = nb.load(img_file)
    mask = nb.load(label_file)
    new_affine = mask.affine
    saveaffine = new_affine
    # new_affine = np.eye(3) * 0.5
    # for i in range(3):
    #     if mask.affine[i, i] < 0:
    #         new_affine[i, i] *= -1
    #
    # saveaffine = np.eye(4)
    # saveaffine[0:3, 0:3] = new_affine
    mask_resample = resample_img(mask, target_affine=new_affine, interpolation='nearest')
    img_resample = resample_img(img, target_affine=new_affine, interpolation='linear')

    mask_np = mask_resample.get_fdata()
    img_np = img_resample.get_fdata()
    #hole fill
    mask_np = create_nonzero_mask(mask_np)
    img_new = nb.Nifti1Image(img_np, saveaffine)
    mask_new = nb.Nifti1Image(mask_np, saveaffine)
    save_img = os.path.join(img_save, idi)
    save_label=os.path.join(vessel_save,idi)
    #save img and mask after affine
    nb.save(img_new,save_img)
    nb.save(mask_new,save_label)

    mask_np[mask_np > 0] = 1
    mask_np = mask_np.astype(np.uint8)

    labels, n = label(mask_np, return_num=True)
    # print('Number of region in mask:', n)

    skl = skeletonize_3d(mask_np)
    labels, n = label(skl, return_num=True)
    # print('Number of region in skeleton:', n)

    skl_nii = nb.Nifti1Image(skl, affine=saveaffine)
    # nb.save(skl_nii,'E:/Test/processed/01_skl.nii.gz')

    # each point has 12 directions
    dir_map = np.zeros(mask_np.shape)
    # print(dir_map.shape)

    # define directions collection(笛卡尔坐标系)
    theta_div = 32
    beta_div = 32
    dir_collections = np.zeros((theta_div * beta_div, 3))
    # print('dir',dir_collections.shape)
    for i in range(theta_div):
        for j in range(beta_div):
            theta = 2 * math.pi * i / theta_div
            beta = 2 * math.pi * j / beta_div
            x = math.cos(theta) * math.cos(beta)
            y = math.cos(theta) * math.sin(beta)
            z = math.sin(theta)
            dir_collections[i * theta_div + j, :] = [x, y, z]
    # print('dir',dir_collections.shape)


    def find_direction_fit(dir_collections, dir):
        # find one direction in dir_collections that is aligned closest with current dir (cosine angle smallest)
        min_cosine = 100
        selected_idx = 0
        for i in range(len(dir_collections)):
            cosine = np.dot(dir_collections[i], dir)
            # cosine = cosine / np.linalg.norm(dir_collections[i]) / np.linalg.norm(dir)
            cosine = np.abs(cosine)
            if cosine < min_cosine:
                min_cosine = cosine
                selected_idx = i
        return selected_idx


    branch_count = 0
    endpoint_count = 0

    # loop through each point in skeleton
    skl_idx = np.argwhere(skl > 0)

    for n in range(skl_idx.shape[0]):
        # current skeleton point coordinate
        x, y, z = skl_idx[n, :]

        # find two nearest points within 3x3x3 volume centered at the current skeleton point
        pts = []
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                for k in range(z - 1, z + 2):
                    if skl[i, j, k] > 0 and ((i != x) or (j != y) or (k != z)):
                        pts.append(np.array([i, j, k]))
        if len(pts) == 2:
            # this is a straight lumen (two direction)
            dir =  pts[1]-pts[0]
            print('dir',dir)
            k = find_direction_fit(dir_collections, dir)
            dir_map[x, y, z] = k
        elif len(pts) < 2:
            # possibly starting point or end point
            # print('Endpoint')
            dir_map[x, y, z] = 1025
            endpoint_count += 1
        elif len(pts) > 2:
            # possibly branching point
            # print('Branch')
            dir = np.array([0, 0, 0])
            dir_map[x, y, z] = 1026
            branch_count += 1

    # in case you want to fill up points on the mask other than centerline
    # for a point not on centerline, its direction is defined by the direction of a cloest point on centerline

    dir_map_full = np.zeros(dir_map.shape)
    from scipy.spatial import distance
    mask_idx = np.argwhere(mask_np > 0)
    dist_map = distance.cdist(mask_idx, skl_idx)
    closest_idx = np.argmin(dist_map, axis=1)
    # print(closest_idx.max())

    for i in range(mask_idx.shape[0]):
        x, y, z = mask_idx[i, :]
        if skl[x, y, z] == 0:  # point not on centerline
            id = closest_idx[i]
            # print(id)
            cx, cy, cz = skl_idx[id, :]  # coordinate of the cloeset point on centerline
            dir_map_full[x, y, z] = dir_map[cx, cy, cz]

    print(dir_map.max(),dir_map.min())
    print(dir_map_full.max(),dir_map_full.min())
    print('endpoint & branch:', endpoint_count, branch_count)
    dir_nii = nb.Nifti1Image(dir_map, affine=saveaffine)
    nb.save(dir_nii, 'E:/Test/processed/01_dir.nii.gz')

    dirfull_nii = nb.Nifti1Image(dir_map_full, affine=saveaffine)
    save_or=os.path.join(oriention_save,idi)
    nb.save(dirfull_nii,save_or)
    print(idi,'is done!')
