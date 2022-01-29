import argparse
import PIL.Image as Image
from PIL import ImageChops
import numpy as np
import os
import random
import copy
import glob
from images_alignment import align_images, trim_image
from tqdm import tqdm

inspect_extension = "inspected_image.tif"
reference_extension = "reference_image.tif"
gt_mask_extension = "inspected_image_mask.png"


def mask_augmentation(err_mask, args):
    # 1-2 bits rotation 90 degrees
    # 3rd bit -> horizontal flip
    aug = random.randrange(8)
    err_mask = err_mask.rotate(90 * (aug%4))
    if aug & 0x04:
        err_mask = err_mask.transpose(Image.FLIP_LEFT_RIGHT)

    deg = random.randrange(360)
    err_mask = err_mask.rotate(deg)

    mn = np.min(np.where(np.asarray(err_mask)[...,0] > 0), axis=1)[:2]
    mx = np.max(np.where(np.asarray(err_mask)[...,0] > 0), axis=1)[:2]
    x = random.randrange(-mn[0],args.mask_size-mx[0]-1)
    y = random.randrange(-mn[1],args.mask_size-mx[1]-1)
    err_mask = ImageChops.offset(err_mask, y, x)

    return err_mask


def shift_reference(image, reference_image, position):
    n = image.shape[0]
    delta = (random.randrange(-1, 2), random.randrange(-1, 2))
    new_position = [position[0] + delta[0], position[1] + delta[1]]
    if not 0 <= new_position[0] < reference_image.shape[0]-n:
        new_position[0] = 0 if new_position[0] < 0 else reference_image.shape[0]-n
    if not 0 <= new_position[1] < reference_image.shape[1]-n:
        new_position[1] = 0 if new_position[1] < 0 else reference_image.shape[1]-n

    image[..., 1] = reference_image[new_position[0]:new_position[0]+n, new_position[1]:new_position[1]+n]
    return image


def reference_modification(img):
    with np.nditer(img[..., 0:1], op_flags=['readwrite']) as it:
        for x in it:
            a = int(x)
            a += random.randrange(-4, 5)
            if not 0 <= a <= 255:
                a = 0 if a < 0 else 255
            x[...] = a
    return img


def add_errors_to_gt_image(img, args):
    global num_of_defect_masks

    if np.sum(img[..., 2]) > 0:  # don't add a real defect to the training data
        return img

    reference_modification(img)

    err_num = random.randrange(num_of_defect_masks)
    mask_fn = os.path.join(args.masks_folder, "%d.png" % err_num)
    err_mask = Image.open(mask_fn)
    err_mask = np.asarray(mask_augmentation(err_mask, args))[..., 0]

    add = random.choice([-1, 1]) * random.randrange(args.low_intensity,args.high_intensity)
    img_layer = img[..., 0].astype(np.float32)
    img_layer[err_mask > 0] += add
    img_layer[img_layer < 0] = 0
    img_layer[img_layer > 255] = 255
    img_layer = img_layer.astype(np.uint8)
    img[..., 0] = img_layer

    # filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.
    # img_layer = signal.convolve2d(img_layer, filter)[1:65, 1:65].astype(np.uint8)
    # np.putmask(img_layer, err_mask==0, img[...,0])
    err_mask[err_mask>0] = 1
    img[..., 2] = err_mask

    return img


def generate_gt_from_example_image(args, inspect_fn, reference_fn, gt_mask_fn):
    example_image = np.asarray(Image.open(inspect_fn))
    reference_image = np.asarray(Image.open(reference_fn))
    mask_img = np.asarray(Image.open(gt_mask_fn))[:, :, :3]
    case = os.path.basename(inspect_fn).split('_')[0]

    delta = align_images(example_image, reference_image)
    example_image_trimmed = trim_image(example_image, delta, first_image=True)
    reference_image_trimmed = trim_image(reference_image, delta, first_image=False)
    mask_img_trimmed = trim_image(mask_img[..., 0], delta, first_image=True)

    full_image = np.dstack((example_image_trimmed, reference_image_trimmed, mask_img_trimmed))

    siz = full_image.shape

    total_i = (siz[0] - args.mask_size) // args.step
    for i in tqdm(range(0, siz[0] - args.mask_size, args.step), total=total_i, desc=f'Rows ', unit=' rows'):
        for j in range(0, siz[1] - args.mask_size, args.step):
            cur_img = copy.deepcopy(full_image[i:i + args.mask_size, j:j + args.mask_size, :])
            if np.sum(cur_img[:, :, 2]) > 0:
                continue

            cur_img = shift_reference(cur_img, reference_image_trimmed, (i, j))
            cur_img = add_errors_to_gt_image(cur_img, args)

            cur_inspected_fn = os.path.join(args.output_folder, "imgs", case + "_%03d_%03d.png" % (i, j))
            cur_mask_fn = os.path.join(args.output_folder, "masks", (case + "_%03d_%03d.png" % (i, j)))
            Image.fromarray(cur_img[:, :, 2]).save(cur_mask_fn)
            cur_img[:, :, 2] = 0
            Image.fromarray(cur_img).save(cur_inspected_fn)

    return


def get_args():
    parser = argparse.ArgumentParser(description='Generate training data from input images')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Cases names (file names will be {case} + "_inspecetd_image.tif" and  {case}+"_reference_image.tif"', required=True)
    parser.add_argument('--input-folder', '-if', metavar='in_folder', type = str, default="./data/examples", help='Input folder contain both inspected and reference images in .tif format')
    parser.add_argument('--output-folder', '-o', metavar='out_folder', type = str, default="./data/gt", help='Output folder (images will be created in {folder} / "imgs" and {folder} / "masks"')
    parser.add_argument('--masks-folder', '-mf', metavar='masks_folder', type = str, default="./data/defect_masks", help='Defect masks folder for GT modification')
    parser.add_argument('--mask-size', '-ms', metavar='size', type=int, default=64, help='Mask size (The window size that the model will work on)')
    parser.add_argument('--seed', '-s', type=int, default=3, help='Seed for the random (-1 to use time)')
    parser.add_argument('--step', '-st', type=int, default=4, help='Step difference between two images taken')
    parser.add_argument('--low-intensity', '-li', type=int, default=15, help='Min intensity of error')
    parser.add_argument('--high-intensity', '-hi', type=int, default=50, help='Max intensity of error')


    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    seed = args.seed if args.seed >= 0 else None
    random.seed(seed)
    global num_of_defect_masks
    num_of_defect_masks = len(glob.glob(os.path.join(args.masks_folder, "*.png")))

    inspect_fp = os.path.join(args.input_folder, "%s_" + inspect_extension)
    reference_fp = os.path.join(args.input_folder, "%s_" + reference_extension)
    gt_mask_fp = os.path.join(args.input_folder, "%s_" + gt_mask_extension)

    for case in args.input:
        print("Now in case: %s" % case)
        generate_gt_from_example_image(args, inspect_fp % case, reference_fp % case, gt_mask_fp % case)


