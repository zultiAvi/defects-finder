import argparse
import logging
import os
from predict import predict_img

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from images_alignment import align_images, trim_image


def predict_img_all_probablilities(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=1.0):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    o = np.zeros((64,64,2))

    with torch.no_grad():
        output = net(img)


        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()
        # F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
        o = np.moveaxis(full_mask.numpy(), [1, 2, 0], [0, 1, 2])
    return o


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=2) * 255).astype(np.uint8))


def predict_section(full_image, pos, io_mask, model, out_threshold, device, debug_folder):
    r, c = pos
    cur_img = full_image[r:r + 64, c:c + 64]
    cur_mask = predict_img(net=model,
                           full_img=Image.fromarray(cur_img),
                           scale_factor=1.0,
                           out_threshold=out_threshold,
                           device=device)

    # mask[r:r+64, c:c+64,0] = np.max((cur_mask[:,:,0], mask[r:r+64, c:c+64,0]), axis=0)
    # mask[r:r+64, c:c+64,1] = np.max((cur_mask[:,:,1], mask[r:r+64, c:c+64,1]), axis=0)
    # mask[r:r+64, c:c+64,2] = np.ones((64, 64))
    cur_mask = (np.argmax(cur_mask, axis=0) * 255 / cur_mask.shape[0]).astype(np.uint8)
    io_mask[r:r + 64, c:c + 64] = np.logical_or(io_mask[r:r + 64, c:c + 64], cur_mask)

    if debug_folder:
        Image.fromarray(cur_img[...,0]).save(os.path.join(debug_folder, "%s_%03d_%03d_inspect.png" % (filename, r, c)))
        Image.fromarray(cur_img[...,1]).save(os.path.join(debug_folder, "%s_%03d_%03d_reference.png" % (filename, r, c)))
        Image.fromarray(cur_mask).save(os.path.join(debug_folder, "%s_%03d_%03d_out.png" % (filename, r, c)))

    return io_mask


def predict_big_image(model, full_img, delta, args, device):
    out_threshold = args.mask_threshold
    R, C = full_img.shape[:2]
    mask = np.zeros((R, C), dtype=np.uint8)
    for r in range(0, R-64, 32):
        for c in range(0, C-64, 32):
            mask = predict_section(full_img, (r, c), mask, model, out_threshold, device, args.debug_folder)
        mask = predict_section(full_img, (r, C-64), mask, model, out_threshold, device, args.debug_folder)

    for c in range(0, C - 64, 32):
        mask = predict_section(full_img, (R-64, c), mask, model, out_threshold, device, args.debug_folder)
    mask = predict_section(full_img, (R-64, C-64), mask, model, out_threshold, device, args.debug_folder)

    # mask[mask[:,:,2]==0] = 1
    # mask[:,:,0] /= mask[:,:,2]
    # mask[:,:,1] /= mask[:,:,2]
    new_R, new_C = R + abs(delta[0]), C + abs(delta[1])
    new_mask = np.zeros((new_R, new_C), dtype=np.uint8)
    dx_s = 0 if delta[0] > 0 else -delta[0]
    dy_s = 0 if delta[1] > 0 else -delta[1]
    new_mask[dx_s:dx_s+R, dy_s:dy_s+C] = mask
    return new_mask


def predict(model, image_for_prediction, delta, out_filename, args, device):
    mask = predict_big_image(model=model,
                             full_img=image_for_prediction,
                             delta=delta,
                             args=args,
                             device=device)

    if not args.no_save:
        result = mask_to_image(mask)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')

    if args.viz:
        logging.info(f'Visualizing results for image {out_filename}, close to continue...')
        plot_img_and_mask(img, mask)


def get_full_image(inspect_fn, reference_fn):
    example_image = np.asarray(Image.open(inspect_fn))
    reference_image = np.asarray(Image.open(reference_fn))

    delta = align_images(example_image, reference_image)
    example_image_trimmed = trim_image(example_image, delta, first_image=True)
    reference_image_trimmed = trim_image(reference_image, delta, first_image=False)
    mask_img = np.zeros(example_image_trimmed.shape)

    full_image = np.dstack((example_image_trimmed, reference_image_trimmed, mask_img)).astype(np.uint8)

    return full_image, delta


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./models/model.pth', metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Cases names (file names will be {case} + "_inspecetd_image.tif" and  {case}+"_reference_image.tif"', required=True)
    parser.add_argument('--input-folder', '-if', metavar='in_folder', type = str, default="./data/examples", help='Input folder for images')
    parser.add_argument('--output-folder', '-of', metavar='out_folder', type = str, default="./data/results", help='Output folder for the result detection mask')
    parser.add_argument('--mask-size', '-ms', metavar='size', type=int, default=64, help='Size of mask')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--debug-folder', '-d', type=str, default="", help='Debug folder, to write temporary small images.')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    inspect_fp = os.path.join(args.input_folder, "%s_inspected_image.tif")
    reference_fp = os.path.join(args.input_folder, "%s_reference_image.tif")
    out_fp = os.path.join(args.output_folder, "%s_detection_mask.png")

    for i, filename in enumerate(args.input):
        logging.info(f'\nPredicting image {filename} ...')
        print("Now in case: %s - %d/%d" % (filename, i+1, len(args.input)))
        img, delta = get_full_image(inspect_fp % filename, reference_fp % filename)

        predict(net, img, delta, out_fp % filename, args, device)
