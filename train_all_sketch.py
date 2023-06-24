import argparse

import os
import re

import numpy as np
from PIL import Image
from skimage import color, io
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable

# from ColorEncoder import ColorEncoder
from models import ColorEncoder, ColorUNet
from vgg_model import vgg19
from discriminator import Discriminator
# from data.data_loader import MultiResolutionDataset
from data.data_loader_sketch import MultiResolutionDataset

from utils import tensor_lab2rgb

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
)


def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def Lab2RGB_out(img_lab):
    img_lab = img_lab.detach().cpu()
    img_l = img_lab[:, :1, :, :]
    img_ab = img_lab[:, 1:, :, :]
    # print(torch.max(img_l), torch.min(img_l))
    # print(torch.max(img_ab), torch.min(img_ab))
    img_l = img_l + 50
    pred_lab = torch.cat((img_l, img_ab), 1)[0, ...].numpy()
    # grid_lab = utils.make_grid(pred_lab, nrow=1).numpy().astype("float64")
    # print(grid_lab.shape)
    out = (np.clip(color.lab2rgb(pred_lab.transpose(1, 2, 0)), 0, 1) * 255).astype("uint8")
    return out


def RGB2Lab(inputs):
    # input [0, 255] uint8
    # out l: [0, 100], ab: [-110, 110], float32
    return color.rgb2lab(inputs)


def Normalize(inputs):
    l = inputs[:, :, 0:1]
    ab = inputs[:, :, 1:3]
    l = l - 50
    lab = np.concatenate((l, ab), 2)

    return lab.astype('float32')


def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2, 0, 1))
    return out


def tensor2numpy(inputs):
    out = inputs[0, ...].detach().cpu().numpy().transpose(1, 2, 0)
    return out


def preprocessing(inputs):
    # input: rgb, [0, 255], uint8
    img_lab = Normalize(RGB2Lab(inputs))
    img = np.array(inputs, 'float32')  # [0, 255]
    img = numpy2tensor(img)
    img_lab = numpy2tensor(img_lab)
    return img.unsqueeze(0), img_lab.unsqueeze(0)


def uncenter_l(inputs):
    l = inputs[:, :1, :, :] + 50
    ab = inputs[:, 1:, :, :]
    return torch.cat((l, ab), 1)


def train(
        args,
        loader,
        colorEncoder,
        colorUNet,
        discriminator,
        vggnet,
        g_optim,
        d_optim,
        device,
):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    g_loss_val = 0
    loss_dict = {}
    recon_val_all = 0
    fea_val_all = 0
    disc_val_all = 0
    disc_val_GAN_all = 0
    disc_val = 0
    count = 0
    criterion_GAN = torch.nn.MSELoss().to(device)

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, args.size // 2 ** 4, args.size // 2 ** 4)
    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    colorEncoder_module = colorEncoder
    colorUNet_module = colorUNet

    for idx in pbar:
        i = idx + args.start_iter + 1

        if i > args.iter:
            print("Done!")

            break

        # img, img_ref, img_lab = next(loader)
        img, img_ref, img_lab, img_lab_sketch = next(loader)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((img.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((img.size(0), *patch))), requires_grad=False)
        # ima =  img_ref.numpy()
        # ima = ima[0].astype('uint8')
        # ima =  Image.fromarray(ima.transpose(1,2,0))
        # ima.show()

        img = img.to(device)  # GT [B, 3, 256, 256]
        img_lab = img_lab.to(device)  # GT
        img_lab_sketch = img_lab_sketch.to(device)

        img_ref = img_ref.to(device)  # tps_transformed image RGB [B, 3, 256, 256]

        img_l = img_lab_sketch[:, :1, :, :] / 50  # [-1, 1] target L
        img_ab = img_lab[:, 1:, :, :] / 110  # [-1, 1] target ab
        # img_ref_ab = img_ref_lab[:,1:,:,:] / 110 # [-1, 1] ref ab

        colorEncoder.train()
        colorUNet.train()
        discriminator.train()

        requires_grad(colorEncoder, True)
        requires_grad(colorUNet, True)
        requires_grad(discriminator, True)

        # ------------------
        #  Train Generators
        # ------------------

        ref_color_vector = colorEncoder(img_ref / 255.)

        fake_swap_ab = colorUNet((img_l, ref_color_vector))  # [-1, 1]

        ## recon l1 loss
        recon_loss = (F.smooth_l1_loss(fake_swap_ab, img_ab))

        ## feature loss
        real_img_rgb = img / 255.
        features_A = vggnet(real_img_rgb, layer_name='all')

        fake_swap_rgb = tensor_lab2rgb(torch.cat((img_l * 50 + 50, fake_swap_ab * 110), 1))  # [0, 1]
        features_B = vggnet(fake_swap_rgb, layer_name='all')
        # fea_loss = F.l1_loss(features_A[-1], features_B[-1]) * 0.1
        # fea_loss = 0

        fea_loss1 = F.l1_loss(features_A[0], features_B[0]) / 32 * 0.1
        fea_loss2 = F.l1_loss(features_A[1], features_B[1]) / 16 * 0.1
        fea_loss3 = F.l1_loss(features_A[2], features_B[2]) / 8 * 0.1
        fea_loss4 = F.l1_loss(features_A[3], features_B[3]) / 4 * 0.1
        fea_loss5 = F.l1_loss(features_A[4], features_B[4]) * 0.1

        fea_loss = fea_loss1 + fea_loss2 + fea_loss3 + fea_loss4 + fea_loss5

        ## discriminator loss
        real_img_rgb = img / 255.
        img_ref_rgb = img_ref / 255.
        zero_ab_image = torch.zeros_like(fake_swap_ab)
        input_img_rgb = tensor_lab2rgb(torch.cat((img_l * 50 + 50, zero_ab_image), 1))  # [0, 1]

        # ima = input_img_rgb.cpu()
        # ima =  ima.numpy()*255
        # ima = ima[0].astype('uint8')
        # ima =  Image.fromarray(ima.transpose(1,2,0))
        # ima.show()

        pred_fake = discriminator(fake_swap_rgb, input_img_rgb, img_ref_rgb)
        disc_loss_GAN = criterion_GAN(pred_fake, valid)
        disc_loss_GAN = disc_loss_GAN * 0.01

        loss_dict["recon"] = recon_loss

        loss_dict["fea"] = fea_loss

        loss_dict["disc_loss_GAN"] = disc_loss_GAN

        g_optim.zero_grad()
        (recon_loss + fea_loss + disc_loss_GAN).backward()
        g_optim.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # if the disc_loss_GAN<0.003, then start to train Discriminator
        if i % 35 == 0:
            # Real loss
            pred_real = discriminator(real_img_rgb, input_img_rgb, img_ref_rgb)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_swap_rgb.detach(), input_img_rgb, img_ref_rgb)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            disc_loss = 0.5 * (loss_real + loss_fake)

            d_optim.zero_grad()
            disc_loss.backward()
            d_optim.step()

            # loss for discriminator itself
            disc_val = disc_loss.mean().item()
            disc_val_all += disc_val
            count += 1

        # --------------
        #  Log Progress
        # --------------

        loss_reduced = reduce_loss_dict(loss_dict)

        recon_val = loss_reduced["recon"].mean().item()
        recon_val_all += recon_val
        # recon_val = 0
        fea_val = loss_reduced["fea"].mean().item()
        fea_val_all += fea_val
        # fea_val = 0

        # loss for generator
        disc_val_GAN = loss_reduced["disc_loss_GAN"].mean().item()
        disc_val_GAN_all += disc_val_GAN

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"recon:{recon_val:.4f}; fea:{fea_val:.4f}; disc_GAN:{disc_val_GAN:.4f}; discriminator:{disc_val:.4f};"
                )
            )

            if i % 100 == 0:
                if disc_val_all != 0:
                    disc_val_all = disc_val_all / count
                print(
                    f"recon_all:{recon_val_all / 100:.4f}; fea_all:{fea_val_all / 100:.4f}; disc_GAN_all:{disc_val_GAN_all / 100:.4f};discriminator:{disc_val_all:.4f};")
                recon_val_all = 0
                fea_val_all = 0
                disc_val_GAN_all = 0
                disc_val_all = 0
                count = 0

            # this code is for model validation, you should prepare you own val dataset and edit code to use it
            # if i % 250 == 0:
            #     with torch.no_grad():
            #         colorEncoder.eval()
            #         colorUNet.eval()
            #
            #         imgsize = 256
            #         for inum in range(12):
            #             val_img_path = 'test_datasets/val_Sketch/in%d.jpg' % (inum + 1)
            #             val_ref_path = 'test_datasets/val_Sketch/ref%d.jpg' % (inum + 1)
            #             # val_img_path = 'test_datasets/val_daytime/day_sample/in%d.jpg'%(inum+1)
            #             # val_ref_path = 'test_datasets/val_daytime/night_sample/dark4.jpg'
            #             out_name = 'in%d_ref%d.png' % (inum + 1, inum + 1)
            #             val_img = Image.open(val_img_path).convert("RGB").resize((imgsize, imgsize))
            #             val_img_ref = Image.open(val_ref_path).convert("RGB").resize((imgsize, imgsize))
            #             val_img, val_img_lab = preprocessing(val_img)
            #             val_img_ref, val_img_ref_lab = preprocessing(val_img_ref)
            #
            #             # val_img = val_img.to(device)
            #             val_img_lab = val_img_lab.to(device)
            #             val_img_ref = val_img_ref.to(device)
            #             # val_img_ref_lab = val_img_ref_lab.to(device)
            #
            #             val_img_l = val_img_lab[:, :1, :, :] / 50.  # [-1, 1]
            #             # val_img_ref_ab = val_img_ref_lab[:,1:,:,:] / 110. # [-1, 1]
            #
            #             ref_color_vector = colorEncoder(val_img_ref / 255.)  # [0, 1]
            #             fake_swap_ab = colorUNet((val_img_l, ref_color_vector))
            #
            #             fake_img = torch.cat((val_img_l * 50, fake_swap_ab * 110), 1)
            #
            #             sample = np.concatenate(
            #                 (tensor2numpy(val_img), tensor2numpy(val_img_ref), Lab2RGB_out(fake_img)), 1)
            #
            #             out_dir = 'training_logs/%s/%06d' % (args.experiment_name, i)
            #             mkdirss(out_dir)
            #             io.imsave('%s/%s' % (out_dir, out_name), sample.astype('uint8'))
            #             torch.cuda.empty_cache()
            if i % 2000 == 0:
                out_dir_g = "experiments/%s" % (args.experiment_name)
                mkdirss(out_dir_g)
                torch.save(
                    {
                        "colorEncoder": colorEncoder_module.state_dict(),
                        "colorUNet": colorUNet_module.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "args": args,
                    },
                    f"%s/{str(i).zfill(6)}_sketch.pt" % (out_dir_g),
                )
                out_dir_d = "experiments/Discriminator"
                mkdirss(out_dir_d)
                torch.save(
                    {
                        "discriminator": discriminator.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"%s/{str(i).zfill(6)}_d.pt" % (out_dir_d),
                )


if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--iter", type=int, default=200000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt_disc", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_disc", type=float, default=0.0002)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    args.start_iter = 0

    vggnet = vgg19(pretrained_path='./experiments/VGG19/vgg19-dcbb9e9d.pth', require_grad=False)
    vggnet = vggnet.to(device)
    vggnet.eval()

    colorEncoder = ColorEncoder(color_dim=512).to(device)
    colorUNet = ColorUNet(bilinear=True).to(device)
    discriminator = Discriminator(in_channels=3).to(device)

    g_optim = optim.Adam(
        list(colorEncoder.parameters()) + list(colorUNet.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr_disc,
        betas=(0.5, 0.999),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            match = re.search(r'\d+', ckpt_name)
            if match:
                args.start_iter = int(match.group(0))
            else:
                args.start_iter = 0
        except ValueError:
            pass

        colorEncoder.load_state_dict(ckpt["colorEncoder"])
        colorUNet.load_state_dict(ckpt["colorUNet"])
        g_optim.load_state_dict(ckpt["g_optim"])

    if args.ckpt_disc is not None:
        print("load discriminator model:", args.ckpt_disc)

        ckpt_disc = torch.load(args.ckpt_disc, map_location=lambda storage, loc: storage)
        discriminator.load_state_dict(ckpt_disc["discriminator"])
        d_optim.load_state_dict(ckpt_disc["d_optim"])
    # print(args.distributed)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-90, 90))
        ]
    )

    datasets = []
    dataset = MultiResolutionDataset(args.datasets, transform, args.size)
    datasets.append(dataset)

    loader = data.DataLoader(
        data.ConcatDataset(datasets),
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    train(
        args,
        loader,
        colorEncoder,
        colorUNet,
        discriminator,
        vggnet,
        g_optim,
        d_optim,
        device,
    )
