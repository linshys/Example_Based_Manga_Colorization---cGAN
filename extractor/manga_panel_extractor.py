# stdlib
import argparse
from argparse import RawTextHelpFormatter
import os
from os.path import splitext, basename, exists, join
from os import makedirs
# 3p
from tqdm import tqdm
import numpy as np
from skimage import measure
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
# project
from utils import get_files, load_image
from skimage import io


class PanelExtractor:
    def __init__(self, min_pct_panel=2, max_pct_panel=90, paper_th=0.35):
        assert min_pct_panel < max_pct_panel, "Minimum percentage must be smaller than maximum percentage"
        self.min_panel = min_pct_panel / 100
        self.max_panel = max_pct_panel / 100
        self.paper_th = paper_th

    def _generate_panel_blocks(self, img):
        img = img if len(img.shape) == 2 else img[:, :, 0]
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY)[1]
        cv2.rectangle(thresh, (0, 0), tuple(img.shape[::-1]), (0, 0, 0), 10)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        ind = np.argsort(stats[:, 4], )[::-1][1]
        panel_block_mask = ((labels == ind) * 255).astype("uint8")
        # Image.fromarray(panel_block_mask).show()
        return panel_block_mask

    def generate_panels(self, img):
        block_mask = self._generate_panel_blocks(img)
        cv2.rectangle(block_mask, (0, 0), tuple(block_mask.shape[::-1]), (255, 255, 255), 10)
        # Image.fromarray(block_mask).show()

        # detect contours
        contours, hierarchy = cv2.findContours(block_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        panels = []
        masks = []
        panel_masks = []
        # print(len(contours))

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            img_area = img.shape[0] * img.shape[1]

            # if the contour is very small or very big, it's likely wrongly detected
            if area < (self.min_panel * img_area) or area > (self.max_panel * img_area):
                continue

            x, y, w, h = cv2.boundingRect(contours[i])
            masks.append(cv2.boundingRect(contours[i]))
            # create panel mask
            panel_mask = np.ones_like(block_mask, "int32")
            cv2.fillPoly(panel_mask, [contours[i].astype("int32")], color=(0, 0, 0))
            # Image.fromarray(panel_mask).show()
            panel_mask = panel_mask[y:y + h, x:x + w].copy()
            # Image.fromarray(panel_mask).show()

            # apply panel mask
            panel = img[y:y + h, x:x + w].copy()
            # Image.fromarray(panel).show()
            panel[panel_mask == 1] = 255
            # Image.fromarray(panel).show()

            panels.append(panel)
            panel_masks.append(panel_mask)

        return panels, masks, panel_masks

    def extract(self, folder):
        print("Loading images ... ", end="")
        # image_list, _, _ = get_files(folder)
        image_list = []
        image_list.append(folder)
        imgs = [load_image(x) for x in image_list]
        print("Done!")

        folder = os.path.dirname(folder)
        # create panels dir
        if not exists(join(folder, "panels")):
            makedirs(join(folder, "panels"))
        folder = join(folder, "panels")

        # remove images with paper texture, not well segmented
        paperless_imgs = []
        for img in tqdm(imgs, desc="Removing images with paper texture"):
            hist, bins = np.histogram(img.copy().ravel(), 256, [0, 256])
            if np.sum(hist[50:200]) / np.sum(hist) < self.paper_th:
                paperless_imgs.append(img)

        if not paperless_imgs:
            return imgs, [], []
        for i, img in tqdm(enumerate(paperless_imgs), desc="extracting panels"):
            panels, masks, panel_masks = self.generate_panels(img)
            name, ext = splitext(basename(image_list[i]))
            for j, panel in enumerate(panels):
                cv2.imwrite(join(folder, f'{name}_{j}.{ext}'), panel)

            # show the order of colorized panels
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('extractor/Open-Sans-Bold.ttf', 160)

            def flatten(l):
                for el in l:
                    if isinstance(el, list):
                        yield from flatten(el)
                    else:
                        yield el

            for i, bbox in enumerate(flatten(masks), start=1):
                w, h = draw.textsize(str(i), font=font)
                y = (bbox[1] + bbox[3] / 2 - h / 2)
                x = (bbox[0] + bbox[2] / 2 - w / 2)
                draw.text((x, y), str(i), (255, 215, 0), font=font)
            img.show()
            return panels, masks, panel_masks

    def concatPanels(self, img_file, fake_imgs, masks, panel_masks):
        img = io.imread(img_file)
        # out_imgs.append(f"D:\MyProject\Python\DL_learning\Manga-Panel-Extractor-master\out\in0_ref0.png")
        # out_imgs.append(f"D:\MyProject\Python\DL_learning\Manga-Panel-Extractor-master\out\in1_ref1.png")
        # out_imgs.append(f"D:\MyProject\Python\DL_learning\Manga-Panel-Extractor-master\out\in2_ref2.png")
        for i in range(len(fake_imgs)):
            x, y, w, h = masks[i]
            # fake_img = io.imread(fake_imgs[i])
            # fake_img = np.array(fake_img)
            fake_img = fake_imgs[i]
            panel_mask = panel_masks[i]
            img[y:y + h, x:x + w][panel_mask == 0] = fake_img[panel_mask == 0]
            # Image.fromarray(img).show()
        out_folder = os.path.dirname(img_file)
        out_name = os.path.basename(img_file)
        out_name = os.path.splitext(out_name)[0]
        out_img_path = os.path.join(out_folder,'color',f'{out_name}_color.png')

        # show image
        Image.fromarray(img).show()
        # save image
        folder_path = os.path.join(out_folder, 'color')
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        io.imsave(out_img_path, img)


def main(args):
    panel_extractor = PanelExtractor(min_pct_panel=args.min_panel, max_pct_panel=args.max_panel)
    panels, masks, panel_masks = panel_extractor.extract(args.folder)
    panel_extractor.concatPanels(args.folder, [], masks, panel_masks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implementation of a Manga Panel Extractor and dialogue bubble text eraser.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-minp", "--min_panel", type=int, choices=range(1, 99), default=5, metavar="[1-99]",
                        help="Percentage of minimum panel area in relation to total page area.")
    parser.add_argument("-maxp", "--max_panel", type=int, choices=range(1, 99), default=90, metavar="[1-99]",
                        help="Percentage of minimum panel area in relation to total page area.")
    parser.add_argument("-f", '--folder', default='./images/002.png', type=str,
                        help="""folder path to input manga pages.
Panels will be saved to a directory named `panels` in this folder.""")

    args = parser.parse_args()
    main(args)
