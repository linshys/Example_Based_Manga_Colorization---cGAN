import argparse

import os

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
from discriminator import Discriminator
from data.data_loader import MultiResolutionDataset

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
        d_optim,
        device,
):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    disc_val_all = 0
    criterion_GAN = torch.nn.MSELoss().to(device)

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, args.size // 2 ** 4, args.size // 2 ** 4)
    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        img, img_ref, img_lab = next(loader)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((img.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((img.size(0), *patch))), requires_grad=False)
        # ima =  img.numpy()
        # ima = ima[0].astype('uint8')
        # ima =  Image.fromarray(ima.transpose(1,2,0))
        # ima.show()

        img = img.to(device)  # GT [B, 3, 256, 256]
        img_lab = img_lab.to(device)  # GT

        img_ref = img_ref.to(device)  # tps_transformed image RGB [B, 3, 256, 256]

        img_l = img_lab[:, :1, :, :] / 50  # [-1, 1] target L
        img_ab = img_lab[:, 1:, :, :] / 110  # [-1, 1] target ab
        # img_ref_ab = img_ref_lab[:,1:,:,:] / 110 # [-1, 1] ref ab

        colorEncoder.eval()
        colorUNet.eval()
        discriminator.train()

        requires_grad(colorEncoder, False)
        requires_grad(colorUNet, False)
        requires_grad(discriminator, True)

        with torch.no_grad():
            ref_color_vector = colorEncoder(img_ref / 255.)
            fake_swap_ab = colorUNet((img_l, ref_color_vector))  # [-1, 1]

        fake_swap_rgb = tensor_lab2rgb(torch.cat((img_l * 50 + 50, fake_swap_ab * 110), 1))  # [0, 1]
        real_img_rgb = img / 255.
        img_ref_rgb = img_ref / 255.

        zero_ab_image = torch.zeros_like(fake_swap_ab)
        input_img_rgb = tensor_lab2rgb(torch.cat((img_l * 50 + 50, zero_ab_image), 1))  # [0, 1]

        # show the gray image

        # input_img_rgb_cpu = input_img_rgb.cpu()
        # ima =  input_img_rgb_cpu.numpy()
        # ima = ima*255
        # ima = ima[0].astype('uint8')
        # ima =  Image.fromarray(ima.transpose(1,2,0))
        # ima.show()

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

        disc_val = disc_loss.mean().item()
        disc_val_all += disc_val

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"discriminator:{disc_val:.4f};"
                )
            )

            if i % 100 == 0:
                print(f"discriminator:{disc_val_all / 100:.4f};")
                disc_val_all = 0
            if i % 1000 == 0:
                out_dir = "experiments/%s" % (args.experiment_name)
                mkdirss(out_dir)
                torch.save(
                    {
                        "discriminator": discriminator.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"%s/{str(i).zfill(6)}_ds.pt" % (out_dir),
                )


if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt_disc", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.start_iter = 0

    colorEncoder = ColorEncoder(color_dim=512).to(device)
    colorUNet = ColorUNet(bilinear=True).to(device)
    discriminator = Discriminator(in_channels=3).to(device)

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        colorEncoder.load_state_dict(ckpt["colorEncoder"])
        colorUNet.load_state_dict(ckpt["colorUNet"])

    if args.ckpt_disc is not None:
        print("load discriminator model:", args.ckpt_disc)

        ckpt_disc = torch.load(args.ckpt_disc, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt_disc)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        discriminator.load_state_dict(ckpt_disc["discriminator"])
        d_optim.load_state_dict(ckpt_disc["d_optim"])

    # print(args.distributed)

    if args.distributed:
        colorEncoder = nn.parallel.DistributedDataParallel(
            colorEncoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        colorUNet = nn.parallel.DistributedDataParallel(
            colorUNet,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 360))
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
        d_optim,
        device,
    )
