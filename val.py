from PIL import Image
from math import log10
import numpy as np
import os

def PSNR(pred, gt):
    pred = pred.clamp(0, 1)
    # pred = (pred - pred.min()) / (pred.max() - pred.min())

    diff = pred - gt
    mse = np.mean(diff.numpy() ** 2)
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse)

def is_image_file(filename):
     return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def main():
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--test_folder1', type=str, default='', help='sr image to use')
    parser.add_argument('--gt_folder', type=str, default='', help='ground truth image to use')
    opt = parser.parse_args()

    image_list1 = sorted([os.path.join(opt.test_folder1, x) for x in os.listdir(opt.test_folder1) if   is_image_file(x)])
    
    psnr = 0.
    for image_name in image_list1:
    	image_path = image_name.split('/')[-1].split('.')[0]
    	print image_path
    	hr = Image.open(os.path.join(gt_folder,image_path))
    	sr = Image.open(image_name)
    	psnr_tmp = PSNR(sr,hr)
    	psnr += psnr_tmp

    psnr = psnr / len(image_list1)

    print psnr

if __name__ == '__main__':
    sys.exit(main())

