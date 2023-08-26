import os

from numpy.core.fromnumeric import shape

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
devicess = [0]
from medpy.io import header, load, save
import time
import argparse
import skimage
from skimage import morphology
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import ZNormalization
from medpy.io import load, save
from tqdm import tqdm
from torchvision import utils
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from scipy.ndimage import distance_transform_edt as distance
import torch.nn.functional as F
from scipy import ndimage
from skimage import data, filters
from numpy.linalg import inv, eig, det
from scipy.ndimage.morphology import binary_dilation
import nibabel as nib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

source_train_dir = hp.source_train_dir
vessel_train_dir = hp.vessel_train_dir
ori_train_dir = hp.oriention_train_dir

source_test_dir = hp.source_test_dir
vessel_test_dir = hp.vessel_test_dir
ori_test_dir = hp.oriention_test_dir

output_dir_test = hp.output_dir_test


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default='logs_CAD', required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default='checkpoint_latest.pt',
                        help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=300, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=3, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=2, help='batch-size')
    training.add_argument('--sample', type=int, default=4, help='number of samples during training')

    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )

    parser.add_argument("--init-lr", type=float, default=0.001, help="learning rate")

    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')

    return parser


def train():
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_train
    os.makedirs(args.output_dir, exist_ok=True)

    if hp.mode == '2d':
        from models.two_d.unet import Unet
        model = Unet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.two_d.miniseg import MiniSeg
        # model = MiniSeg(in_input=hp.in_class, classes=hp.out_class)

        # from models.two_d.fcn import FCN32s as fcn
        # model = fcn(in_class =hp.in_class,n_class=hp.out_class)

        # from models.two_d.segnet import SegNet
        # model = SegNet(input_nbr=hp.in_class,label_nbr=hp.out_class)

        # from models.two_d.deeplab import DeepLabV3
        # model = DeepLabV3(in_class=hp.in_class,class_num=hp.out_class)

        # from models.two_d.unetpp import ResNet34UnetPlus
        # model = ResNet34UnetPlus(num_channels=hp.in_class,num_class=hp.out_class)

        # from models.two_d.pspnet import PSPNet
        # model = PSPNet(in_class=hp.in_class,n_classes=hp.out_class)

    elif hp.mode == '3d':

        #         from models.three_d.unet3d import UNet3D
        #         model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32)

        #         from models.three_d.lambdanet import UNet3D
        #         model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32,class_=hp.class_)
        #两个分支（val最好0.69）
        # from models.three_d.ORNet import UNet3D
        # model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32, class_=hp.class_,ori_class=hp.ori_class)
        #单个分支，三个输出
        from models.three_d.Ori_Net import OriNet
        model = OriNet(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32, class_=hp.class_,
                       ori_class=hp.ori_class)
        # from models.three_d.unet3d import UNet3D
        # model = UNet3D(in_channels=hp.in_class, n_classes=hp.out_class, base_n_filter=32)
        # from models.three_d.residual_unet3d import UNet
        # model = UNet(in_channels=hp.in_class, n_classes=hp.out_class, base_n_filter=32)

        # from models.three_d.fcn3d import FCN_Net
        # model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class)

        # from models.three_d.highresnet import HighRes3DNet
        # model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class)

        # from models.three_d.densenet3d import SkipDenseNet3D
        # model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.vnet3d import VNet
        # model = VNet(in_channels=hp.in_class, classes=hp.out_class)


    model = torch.nn.DataParallel(model, device_ids=devicess)
    from torchsummary import summary
    print('params', summary(model, input_size=(1, 128, 128, 128)))
    inputs = torch.randn(1, 1, 128, 128, 128).to(torch.float32)
    from thop import profile
    flops, params = profile(model, inputs=(inputs.to(device),))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.8)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0
    model.cuda()
    from loss_function import Binary_Loss,BinaryDiceLoss, cross_entropy_3D
    CE_loss = Binary_Loss().cuda()
    Dice_loss = BinaryDiceLoss().cuda()
    writer = SummaryWriter(args.output_dir)
    train_dataset = MedData_train(source_train_dir, vessel_train_dir,ori_train_dir)
    train_loader = DataLoader(train_dataset.queue_dataset,
                              batch_size=args.batch,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True)
    model.train()
    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)
    loss1 = []
    for epoch in range(1, epochs + 1):
        print("epoch:" + str(epoch))
        epoch += elapsed_epochs
        train_epoch_avg_loss = []
        num_iters = 0
        fpr = []
        fnr = []
        dc = []
        dice = []
        f_loss = []
        for i, batch in enumerate(train_loader):
            if hp.debug:
                if i >= 1:
                    break
            optimizer.zero_grad()
            x = batch['img']['data']
            y_lab = batch['vessel']['data']
            y_ori = batch['oriention']['data']
            #半径
            if torch.sum(y_lab) > 0:
                posmask = y_lab.numpy().astype(np.bool_)
                if posmask.any():
                    y_dist = torch.tensor(distance(posmask).astype('int16'))
                y_dist = F.one_hot(y_dist.long(), hp.class_)
                y_dist = y_dist.permute(0, 5, 2, 3, 4, 1)
                y_dist = y_dist.squeeze(5)
                #方向
                y_ori = F.one_hot(y_ori.long(), hp.ori_class)
                y_ori = y_ori.permute(0, 5, 2, 3, 4, 1)
                y_ori = y_ori.squeeze(5)
                #数据送入cuda
                x = x.type(torch.FloatTensor).cuda()
                y_lab = y_lab.type(torch.FloatTensor).cuda()
                y_dist = y_dist.type(torch.FloatTensor).cuda()
                y_ori = y_ori.type(torch.FloatTensor).cuda()
                seg,dist,oriention = model(x)
                # for metrics
                map_logits = torch.sigmoid(seg)
                map_labels = map_logits.clone()
                map_labels[map_labels > 0.5] = 1
                map_labels[map_labels <= 0.5] = 0
                seg_loss=Dice_loss(seg,y_lab)
                dist_loss=CE_loss(dist,y_dist)
                ori_loss = CE_loss(oriention, y_ori)
                loss = seg_loss+dist_loss+ori_loss
                loss1.append(loss)

                num_iters += 1
                loss.backward()

                optimizer.step()
                iteration += 1
                false_positive_rate, false_negtive_rate, dice1 = metric(y_lab.cpu(), map_labels.cpu())
                ## log
                writer.add_scalar('Training/Loss', loss.item(), iteration)
                writer.add_scalar('Training/false_positive_rate', false_positive_rate, iteration)
                writer.add_scalar('Training/false_negtive_rate', false_negtive_rate, iteration)
                writer.add_scalar('Training/dice', dice1, iteration)

                dice.append(dice1)
                f_loss.append(loss.item())

        scheduler.step()
        print(epoch, 'mean dice', np.mean(dice), 'max dice', np.max(dice), 'max dice', np.min(dice))
        print(epoch, 'mean loss', np.mean(f_loss), 'max loss', np.max(f_loss), 'max loss', np.min(f_loss))

        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )

        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:
            torch.save(
                {

                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )
            from data_function import MedData_test

            # train_path = r'H:\dist_seg\logs_CAD'
            # # print(os.path.join(train_path, args.latest_checkpoint_file))
            # ckpt = torch.load(os.path.join(train_path, args.latest_checkpoint_file),
            #                   map_location=lambda storage, loc: storage)
            # model.load_state_dict(ckpt["model"], False)

            model.cuda()

            test_dataset = MedData_test(source_test_dir, vessel_test_dir,ori_test_dir)
            znorm = ZNormalization()
            if hp.mode == '3d':
                patch_overlap = 0, 0, 0
                patch_size = hp.patch_size, hp.patch_size, hp.patch_size
            me = []
            for i, subj in enumerate(test_dataset.subjects):
                subj = znorm(subj)
                grid_sampler = torchio.inference.GridSampler(
                    subj,
                    patch_size,
                    patch_overlap,
                )
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=2)
                aggregator = torchio.inference.GridAggregator(grid_sampler)
                aggregator_1 = torchio.inference.GridAggregator(grid_sampler)
                model.eval()
                with torch.no_grad():
                    for patches_batch in tqdm(patch_loader):
                        input_tensor = patches_batch['img'][torchio.DATA].to(device)
                        y_label = patches_batch['vessel'][torchio.DATA]
                        locations = patches_batch[torchio.LOCATION]
                        seg,dist,oriention = model(input_tensor)
                        map_logits = torch.sigmoid(seg)
                        map_labels = map_logits.clone()
                        map_labels[map_labels > 0.5] = 1
                        map_labels[map_labels <= 0.5] = 0
                        aggregator.add_batch(map_labels, locations)
                        aggregator_1.add_batch(y_label, locations)
                output_tensor = aggregator.get_output_tensor()
                label = aggregator_1.get_output_tensor()
                false_positive_rate, false_negtive_rate, dice1 = metric(output_tensor.cpu(), label.cpu())
                me.append(dice1)
            print('epoch', epoch, 'val', np.mean(me))
    writer.close()

def test():
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_test

    os.makedirs(output_dir_test, exist_ok=True)

    if hp.mode == '2d':
        from models.two_d.unet import Unet
        model = Unet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.two_d.miniseg import MiniSeg
        # model = MiniSeg(in_input=hp.in_class, classes=hp.out_class)

        # from models.two_d.fcn import FCN32s as fcn
        # model = fcn(in_class =hp.in_class,n_class=hp.out_class)

        # from models.two_d.segnet import SegNet
        # model = SegNet(input_nbr=hp.in_class,label_nbr=hp.out_class)

        # from models.two_d.deeplab import DeepLabV3
        # model = DeepLabV3(in_class=hp.in_class,class_num=hp.out_class)

        # from models.two_d.unetpp import ResNet34UnetPlus
        # model = ResNet34UnetPlus(num_channels=hp.in_class,num_class=hp.out_class)

        # from models.two_d.pspnet import PSPNet
        # model = PSPNet(in_class=hp.in_class,n_classes=hp.out_class)

    elif hp.mode == '3d':
        #         from models.three_d.unet3d import UNet
        #         model = UNet(in_channels=hp.in_class, n_classes=hp.out_class, base_n_filter=2)
        # from models.three_d.lambdanet import UNet3D
        # model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32,class_=hp.class_)
        # from models.three_d.fftlambda import UNet3D
        # model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32, class_=hp.class_,
        #                w=hp.patch_size, h=hp.patch_size, d=hp.patch_size)
        from models.three_d.Ori_Net import UNet3D
        model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=32, class_=hp.class_,
                       ori_class=hp.ori_class)

        # from models.three_d.fcn3d import FCN_Net
        # model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class)

        # from models.three_d.highresnet import HighRes3DNet
        # model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class)

        # from models.three_d.densenet3d import SkipDenseNet3D
        # model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.vnet3d import VNet
        # model = VNet(in_channels=hp.in_class, classes=hp.out_class)

    model = torch.nn.DataParallel(model, device_ids=devicess, output_device=[1])

    print("load model:", args.ckpt)

    train_path = 'H:\oriention_seg\logs_CAD'
    print(os.path.join(train_path, args.latest_checkpoint_file))

    ckpt = torch.load(os.path.join(train_path,'checkpoint_0090.pt'), map_location=lambda storage, loc: storage)
    # ckpt = torch.load(train_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"], False)

    model.cuda()

    test_dataset = MedData_test(source_test_dir, vessel_test_dir,ori_test_dir)
    znorm = ZNormalization()

    if hp.mode == '3d':
        patch_overlap = 0, 0, 0
        patch_size = hp.patch_size, hp.patch_size, hp.patch_size
    elif hp.mode == '2d':
        patch_overlap = 4, 4, 0
        patch_size = hp.patch_size, hp.patch_size, 1

    for i, subj in enumerate(test_dataset.subjects):
        subj = znorm(subj)
        grid_sampler = torchio.inference.GridSampler(
            subj,
            patch_size,
            patch_overlap,
        )

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=2)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        aggregator_1 = torchio.inference.GridAggregator(grid_sampler)
        aggregator_2 = torchio.inference.GridAggregator(grid_sampler)
        aggregator_3 = torchio.inference.GridAggregator(grid_sampler)

        model.eval()
        with torch.no_grad():
            for patches_batch in patch_loader:

                input_tensor = patches_batch['img'][torchio.DATA].to(device)
                vessel_tensor= patches_batch['vessel'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]

                seg,dist,oriention = model(input_tensor)
                # print('dist', dist.size(),'oriention',oriention.size())

                #seg
                map_logits = torch.sigmoid(seg)
                map_labels = map_logits.clone()
                map_labels[map_labels > 0.6] = 1
                map_labels[map_labels <= 0.4] = 0
                #dist
                dist_labels = dist.clone()
                dist_labels = torch.argmax(dist_labels, dim=1)
                dist_labels = dist_labels.unsqueeze(1)
                #oriention
                ori_labels = oriention.clone()
                # ori_labels = torch.argmax(ori_labels, dim=1)
                ori_labels=ori_labels.argmax(dim=1)
                ori_labels = ori_labels.unsqueeze(1)
                # print('dist_labels',dist_labels.size(),'ori_labels', ori_labels.size())

                aggregator.add_batch(map_labels, locations)
                aggregator_1.add_batch(dist_labels, locations)
                aggregator_2.add_batch(ori_labels, locations)
                aggregator_3.add_batch(vessel_tensor, locations)

        seg_tensor = aggregator.get_output_tensor()
        dist_tensor = aggregator_1.get_output_tensor()
        ori_tensor = aggregator_2.get_output_tensor()
        ves_tensor = aggregator_3.get_output_tensor()
        affine = subj['img']['affine']

        a = np.array(patches_batch['img']['stem'])
        false_positive_rate, false_negtive_rate, dice1 = metric(seg_tensor, ves_tensor)
        # print('id',a[0],'dice',dice1)
        if (hp.in_class == 1) and (hp.out_class == 1):
            label_save_path=os.path.join(output_dir_test,'vessel')
            if not os.path.exists(label_save_path):
                os.mkdir(label_save_path)
            label_save_file = os.path.join(label_save_path, str(a[0]) + ".nii.gz")
            label_image = torchio.ScalarImage(tensor=seg_tensor.cpu().numpy(), affine=affine)
            label_image.save(label_save_file)

            dist_save_path = os.path.join(output_dir_test, 'dist')
            if not os.path.exists(dist_save_path):
                os.mkdir(dist_save_path)
            dist_save_file = os.path.join(dist_save_path, str(a[0]) + ".nii.gz")
            dist_image = torchio.ScalarImage(tensor=dist_tensor.cpu().numpy(), affine=affine)
            dist_image.save(dist_save_file)

            ori_save_path = os.path.join(output_dir_test, 'oriention')
            if not os.path.exists(ori_save_path):
                os.mkdir(ori_save_path)
            ori_save_file = os.path.join(ori_save_path, str(a[0]) + ".nii.gz")
            ori_image = torchio.ScalarImage(tensor=ori_tensor.cpu().numpy(), affine=affine)
            ori_image.save(ori_save_file)
            # print('it is over!')

        else:
            output_tensor = output_tensor.unsqueeze(1)
            output_tensor_1 = output_tensor_1.unsqueeze(1)

            output_image_artery_float = torchio.ScalarImage(tensor=output_tensor[0].numpy(), affine=affine)
            output_image_artery_float.save(os.path.join(output_dir_test, str(i) + "_result_float_artery.mhd"))

            output_image_artery_int = torchio.ScalarImage(tensor=output_tensor_1[0].numpy(), affine=affine)
            output_image_artery_int.save(os.path.join(output_dir_test, str(i) + "_result_int_artery.mhd"))

            output_image_lung_float = torchio.ScalarImage(tensor=output_tensor[1].numpy(), affine=affine)
            output_image_lung_float.save(os.path.join(output_dir_test, str(i) + "_result_float_lung.mhd"))

            output_image_lung_int = torchio.ScalarImage(tensor=output_tensor_1[1].numpy(), affine=affine)
            output_image_lung_int.save(os.path.join(output_dir_test, str(i) + "_result_int_lung.mhd"))

            output_image_trachea_float = torchio.ScalarImage(tensor=output_tensor[2].numpy(), affine=affine)
            output_image_trachea_float.save(os.path.join(output_dir_test, str(i) + "_result_float_trachea.mhd"))

            output_image_trachea_int = torchio.ScalarImage(tensor=output_tensor_1[2].numpy(), affine=affine)
            output_image_trachea_int.save(os.path.join(output_dir_test, str(i) + "_result_int_trachea.mhd"))

            output_image_vein_float = torchio.ScalarImage(tensor=output_tensor[3].numpy(), affine=affine)
            output_image_vein_float.save(os.path.join(output_dir_test, str(i) + "_result_float_veiny.mhd"))

            output_image_vein_int = torchio.ScalarImage(tensor=output_tensor_1[3].numpy(), affine=affine)
            output_image_vein_int.save(os.path.join(output_dir_test, str(i) + "_result_int_vein.mhd"))


if __name__ == '__main__':
    if hp.train_or_test == 'train':
        train()
    elif hp.train_or_test == 'test':
        test()