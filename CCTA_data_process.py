from SimpleITK.SimpleITK import Image
import numpy as np
import SimpleITK 
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib

def resample_image(itk_image, out_spacing):
    original_spacing = itk_image.GetSpacing()
    
    original_size = itk_image.GetSize()
    

    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]
 
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
 
    resample.SetInterpolator(sitk.sitkBSpline)
 
    return resample.Execute(itk_image)

def resample_label(itk_image, out_spacing):
    original_spacing = itk_image.GetSpacing()
    
    original_size = itk_image.GetSize()
    

    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]
 
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
 
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
 
    return resample.Execute(itk_image)

def connected_domain_2(image, mask=True):
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label +1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    largest_area = area_list[num_list_sorted[0] - 1]
    final_label_list = [num_list_sorted[0]]

    for idx, i in enumerate(num_list_sorted[1:]):
        if area_list[i-1] >= (largest_area//10):
            final_label_list.append(i)
        else:
            break
    output = sitk.GetArrayFromImage(output_ex)

    for one_label in num_list:
        if  one_label in final_label_list:
            continue
        x, y, z, w, h, d = stats.GetBoundingBox(one_label)
        one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
        output[z: z + d, y: y + h, x: x + w] *= one_mask

    if mask:
        output = (output > 0).astype(np.uint8)
    else:
        output = ((output > 0)*255.).astype(np.uint8)
    return output

def load_dicom(path):
    reader = SimpleITK.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(img_names)
    image = reader.Execute()
    spacing = image.GetSpacing()
    # print('image origin spacing',spacing)
    image = resample_image(image,out_spacing=[spacing[0],spacing[1],0.5])
    return image

def load_nii(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    spacing = image.GetSpacing()
#     print('label oringin spacing',spacing)
    image = resample_label(image,out_spacing=[spacing[0],spacing[1],0.5])
    # origin = image.GetOrigin()
    # spacing = image.GetSpacing()
    # direction = image.GetDirection()
    # label_data = sitk.GetArrayFromImage(image)
    # image_sitk = connected_domain_2(label_data, mask=True)
    # image_sitk = sitk.GetImageFromArray(image_sitk)
    # image_sitk.SetOrigin(origin)
    # image_sitk.SetSpacing(spacing)
    # image_sitk.SetDirection(direction)
    return image


if __name__ == '__main__':
    nrrd_path =  '/root/data/vessel_nrrd'
    # label_path = '/root/data/segment_nrrd'
    dicom_path = '/root/data/anoymize'
    save_path = '/root/workspace/DeepcadData/CAD_data/image_raw'
    save_path_process = '/root/workspace/DeepcadData/CAD_data/image'
    vessel_save = '/root/workspace/DeepcadData/CAD_data/label'
    # label_save = '/root/workspace/DeepcadData/CAD_mha/label_p'
   
    for i in os.listdir(nrrd_path):
        print(i)
        # nrrd_path1 = os.path.join(nrrd_path,str(i))
        # nrrd_path2 = os.path.join(nrrd_path1,os.listdir(nrrd_path1)[0])
        # vessel_label = load_nii(nrrd_path2)
        
       
        nrrd_path1 = os.path.join(nrrd_path,str(i))
        nrrd_path2 = os.path.join(nrrd_path1,os.listdir(nrrd_path1)[0])
        nrrd_path3 = os.path.join(nrrd_path2,os.listdir(nrrd_path2)[0])
        nrrd_path4 = os.path.join(nrrd_path3,os.listdir(nrrd_path3)[0])
        vessel_label = load_nii(nrrd_path4)
        
        vessel_save_path = os.path.join(vessel_save,str(i)+".nii.gz") 
        sitk.WriteImage(vessel_label, vessel_save_path)
        reader = sitk.ImageFileReader()
        reader.SetFileName(vessel_save_path)
        vessel = reader.Execute()
        vessel_spacing = vessel.GetSpacing()
       
        
        #保存连着主动脉的label
#         label_path1 = os.path.join(label_path,str(i))
#         label_path2 = os.path.join(label_path1,os.listdir(label_path1)[0])
#         label_path3 = os.path.join(label_path2,os.listdir(label_path2)[0])
#         label_path4 = os.path.join(label_path3,os.listdir(label_path3)[0])
#         label = load_nii(label_path4)
#         label_save_path = os.path.join(label_save,str(i)+".nii.gz") 
#         sitk.WriteImage(label, label_save_path)
#         reader = sitk.ImageFileReader()
#         reader.SetFileName(label_save_path)
#         label = reader.Execute()
#         label_spacing = label.GetSpacing()
        
        #raw 图像
        raw_path = os.path.join(dicom_path,str(i),'DICOM')
        raw_path = os.path.join(dicom_path,str(i),os.listdir(nrrd_path1)[0],os.listdir(nrrd_path2)[0])
        # print(raw_path)
        raw_image = load_dicom(raw_path)
        # print(raw_image)
        raw_save_path = os.path.join(save_path,str(i)+".nii.gz")
        sitk.WriteImage(raw_image, raw_save_path)

        
        
        reader = sitk.ImageFileReader()
        reader.SetFileName(raw_save_path)
        image = reader.Execute()
        image = sitk.Cast(image, sitk.sitkInt16)
        seg1 = (image < (-224))
        morphFilter = sitk.BinaryMorphologicalClosingImageFilter()
        morphFilter.SetKernelRadius(10)
        morphFilter.SetKernelType(sitk.sitkBall)
        morphFilter.SetForegroundValue(1)
        seg2 = morphFilter.Execute(seg1)
        seg3 = 1 - seg2
        seg3 = sitk.Cast(seg3, sitk.sitkInt16)
        img_processed = image * seg3
        img_processed = img_processed * sitk.Cast(img_processed>0, sitk.sitkInt16)
        spacing = img_processed.GetSpacing()
        
        save_change_path3 = os.path.join(save_path_process,str(i)+".nii.gz")
        sitk.WriteImage(img_processed, save_change_path3)
        print(i,'is process done !')