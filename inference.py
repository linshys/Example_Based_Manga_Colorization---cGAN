import os
import numpy as np
from skimage import color, io

import torch
import torch.nn.functional as F

from PIL import Image
from models import ColorEncoder, ColorUNet
from extractor.manga_panel_extractor import PanelExtractor
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Lab2RGB_out(img_lab):
    img_lab = img_lab.detach().cpu()
    img_l = img_lab[:,:1,:,:]
    img_ab = img_lab[:,1:,:,:]
    # print(torch.max(img_l), torch.min(img_l))
    # print(torch.max(img_ab), torch.min(img_ab))
    img_l = img_l + 50
    pred_lab = torch.cat((img_l, img_ab), 1)[0,...].numpy()
    # grid_lab = utils.make_grid(pred_lab, nrow=1).numpy().astype("float64")
    # print(grid_lab.shape)
    out = (np.clip(color.lab2rgb(pred_lab.transpose(1, 2, 0)), 0, 1)* 255).astype("uint8")
    return out

def RGB2Lab(inputs):
    return color.rgb2lab(inputs)

def Normalize(inputs):
    l = inputs[:, :, 0:1]
    ab = inputs[:, :, 1:3]
    l = l - 50
    lab = np.concatenate((l, ab), 2)

    return lab.astype('float32')

def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2,0,1))
    return out

def tensor2numpy(inputs):
    out = inputs[0,...].detach().cpu().numpy().transpose(1,2,0)
    return out

def preprocessing(inputs):
    # input: rgb, [0, 255], uint8
    img_lab = Normalize(RGB2Lab(inputs))
    img = np.array(inputs, 'float32') # [0, 255]
    img = numpy2tensor(img)
    img_lab = numpy2tensor(img_lab)
    return img.unsqueeze(0), img_lab.unsqueeze(0)

if __name__ == "__main__":
    device = "cuda"

    # model_name = 'Color2Manga_sketch'
    ckpt_path = 'experiments/Color2Manga_gray/074000_gray.pt'
    test_dir_path = 'test_datasets/gray_test'
    no_extractor = False
    # imgs_num = len(os.listdir(test_dir_path)) // 2
    imgsize = 256

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default=None, help="path of input image")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=None, help="path of model weight")
    parser.add_argument("-ne", "--no_extractor", action='store_true',
                        help="Do not segment the manga panels.")

    args = parser.parse_args()

    if args.path:
        ckpt_path = args.path
    if args.size:
        imgsize = args.size
    if args.ckpt:
        test_dir_path = args.ckpt
    if args.no_extractor:
        no_extractor = args.no_extractor


    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    colorEncoder = ColorEncoder().to(device)
    colorEncoder.load_state_dict(ckpt["colorEncoder"])
    colorEncoder.eval()

    colorUNet = ColorUNet().to(device)
    colorUNet.load_state_dict(ckpt["colorUNet"])
    colorUNet.eval()

    imgs = []
    imgs_lab = []

    # for i in range(imgs_num):
    # idx = i
    # print('Image', idx, 'Input Image', 'in%d.JPEG'%idx, 'Ref Image', 'ref%d.JPEG'%idx)

    while 1:
        print(f'make sure both manga image and reference images are under this path{test_dir_path}')
        img_path = input("please input the name of image needed to be colorized(with file extension): ")
        img_path = os.path.join(test_dir_path, img_path)
        img_name = os.path.basename(img_path)
        img_name = os.path.splitext(img_name)[0]

        if no_extractor:
            ref_img_path = os.path.join(test_dir_path, input(f"{1}/{1} reference image:"))

            img1 = Image.open(img_path).convert("RGB")
            width, height = img1.size
            img2 = Image.open(ref_img_path).convert("RGB")

            img1, img1_lab = preprocessing(img1)
            img2, img2_lab = preprocessing(img2)

            img1 = img1.to(device)
            img1_lab = img1_lab.to(device)
            img2 = img2.to(device)
            img2_lab = img2_lab.to(device)

            # print('-------',torch.max(img1_lab[:,:1,:,:]), torch.min(img1_lab[:,1:,:,:]))

            with torch.no_grad():
                img2_resize = F.interpolate(img2 / 255., size=(imgsize, imgsize), mode='bilinear',
                                            recompute_scale_factor=False, align_corners=False)
                img1_L_resize = F.interpolate(img1_lab[:, :1, :, :] / 50., size=(imgsize, imgsize), mode='bilinear',
                                              recompute_scale_factor=False, align_corners=False)

                color_vector = colorEncoder(img2_resize)

                fake_ab = colorUNet((img1_L_resize, color_vector))
                fake_ab = F.interpolate(fake_ab * 110, size=(height, width), mode='bilinear',
                                        recompute_scale_factor=False, align_corners=False)

                fake_img = torch.cat((img1_lab[:, :1, :, :], fake_ab), 1)
                fake_img = Lab2RGB_out(fake_img)
                # io.imsave(out_img_path, fake_img)

                out_folder = os.path.dirname(img_path)
                out_name = os.path.basename(img_path)
                out_name = os.path.splitext(out_name)[0]
                out_img_path = os.path.join(out_folder, 'color', f'{out_name}_color.png')

                # show image
                Image.fromarray(fake_img).show()
                # save image
                folder_path = os.path.join(out_folder, 'color')
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                io.imsave(out_img_path, fake_img)

            continue



        # extract panels from manga
        panel_extractor = PanelExtractor(min_pct_panel=5, max_pct_panel=90)
        panels, masks, panel_masks = panel_extractor.extract(img_path)
        panel_num = len(panels)

        ref_img_paths = []
        # ref_img_path = os.path.join(test_dir_path, '%03d_ref.png' % idx)
        print("Please enter the name of the reference image in order according to the number prompts on the picture")
        for i in range(panel_num):
            ref_img_path = os.path.join(test_dir_path, input(f"{i+1}/{panel_num} reference image:"))
            ref_img_paths.append(ref_img_path)




        fake_imgs = []
        for i in range(panel_num):
            img1 = Image.fromarray(panels[i]).convert("RGB")
            width, height = img1.size
            img2 = Image.open(ref_img_paths[i]).convert("RGB")

            # img1 = Image.open(img_path).convert("RGB")
            # width, height = img1.size
            # img2 = Image.open(ref_img_path).convert("RGB")

            img1, img1_lab = preprocessing(img1)
            img2, img2_lab = preprocessing(img2)

            img1 = img1.to(device)
            img1_lab = img1_lab.to(device)
            img2 = img2.to(device)
            img2_lab = img2_lab.to(device)

            # print('-------',torch.max(img1_lab[:,:1,:,:]), torch.min(img1_lab[:,1:,:,:]))

            with torch.no_grad():
                img2_resize = F.interpolate(img2 / 255., size=(imgsize, imgsize), mode='bilinear', recompute_scale_factor=False, align_corners=False)
                img1_L_resize = F.interpolate(img1_lab[:,:1,:,:] / 50., size=(imgsize, imgsize), mode='bilinear', recompute_scale_factor=False, align_corners=False)

                color_vector = colorEncoder(img2_resize)

                fake_ab = colorUNet((img1_L_resize, color_vector))
                fake_ab = F.interpolate(fake_ab*110, size=(height, width), mode='bilinear', recompute_scale_factor=False, align_corners=False)

                fake_img = torch.cat((img1_lab[:,:1,:,:], fake_ab), 1)
                fake_img = Lab2RGB_out(fake_img)
                # io.imsave(f'test_datasets/gray_test/panels/{i}.png', fake_img)
                fake_imgs.append(fake_img)

        if panel_num == 1:
            out_folder = os.path.dirname(img_path)
            out_name = os.path.basename(img_path)
            out_name = os.path.splitext(out_name)[0]
            out_img_path = os.path.join(out_folder,'color',f'{out_name}_color.png')

            # show image
            Image.fromarray(fake_imgs[0]).show()
            # save image
            folder_path = os.path.join(out_folder, 'color')
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            io.imsave(out_img_path, fake_imgs[0])
        else:
            panel_extractor.concatPanels(img_path, fake_imgs, masks, panel_masks)

        print(f'colored image has been put to: {test_dir_path}color')

