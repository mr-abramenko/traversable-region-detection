#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_img_with_tr_regs.py

Объединяет изображение и регион доступный для движения.
"""
import numpy as np
import argparse
import time
import glob
import sys
import os
from matplotlib import pyplot as plt

def overlay_image_with_tr_mask(in_image, mask_tr_regions,
                               vis_channel = 1, threshold = 0.5):
    """
    Объединяет изображение и карту региона доступного для движения.
    """
    visImage = in_image.copy()

    if len(mask_tr_regions.shape) == 3: # if gt_image_2
        mask_tr_regions = mask_tr_regions[:, :, 2]  # blue channel

    channelPart = ( visImage[:, :, vis_channel] *
                   (mask_tr_regions > threshold)- mask_tr_regions
                   )
    channelPart[channelPart < 0] = 0
    visImage[:, :, vis_channel] = ( visImage[:, :, vis_channel] *
                                    (mask_tr_regions <= threshold) +
                                    (mask_tr_regions > threshold) *
                                    mask_tr_regions + channelPart
                                    )
    return visImage

# =============================================================================
# Скипт
# =============================================================================
if __name__ == "__main__":
    start_time = time.clock()

    # Анализ аргументов командной строки
    parser = argparse.ArgumentParser(prog='python show_img_with_tr_regs.py',
                                 description="Show image and traversable regions",
                                 epilog="Abramenko A.A.")
    parser.add_argument('imgl',
                        help="path to the left image(s)",
                        metavar="IMG_L")
    parser.add_argument('trdir',
                        help="path to traversable region map(s)",
                        metavar="TR_DIR")
    parser.add_argument('odir',
                        help="path to output directory",
                        metavar="ODIR")
    parser.add_argument('-v',
                        action='version',
                        version='%(prog)s 1.0.0')
    args = parser.parse_args()

    workdir = os.getcwd()
    imgLpath = os.path.abspath(args.imgl)
    trdirpath = os.path.abspath(args.trdir)
    outdirpath = os.path.abspath(args.odir)

    if not os.path.exists(outdirpath):
        os.makedirs(outdirpath)

    if ( os.path.isfile(imgLpath) and os.path.isfile(trdirpath) ):
        imgLdirpath = os.path.dirname(imgLpath)
        imgTRdirpath = os.path.dirname(trdirpath)

        imgLfilenames = [os.path.basename(imgLpath)]
        imgTRfilenames = [os.path.basename(trdirpath)]
    elif ( os.path.isdir(imgLpath) and os.path.isdir(trdirpath) ):
        imgLdirpath = imgLpath
        imgTRdirpath = trdirpath

        os.chdir(trdirpath)
        imgTRfilenames = (glob.glob('*.png'))
        imgLfilenames = imgTRfilenames
        os.chdir(workdir)
    else:
        print("INFO: UNSUCCESS")
        print("....: invalid path to input files")
        sys.exit(1)

    # Объединение
    for imgLname, imgTRname in zip(imgLfilenames, imgTRfilenames):

        tags = imgTRname.split('_')
        if len(tags) == 3:
            imgLname = tags[0] + "_" + tags[2]
        print("....:", imgLname)

        if ( os.path.exists(f"{imgLdirpath}/{imgLname}")
                and os.path.exists(f"{imgTRdirpath}/{imgTRname}")):
            imgL = plt.imread(f"{imgLdirpath}/{imgLname}")
            imgTR = plt.imread(f"{imgTRdirpath}/{imgTRname}").astype(np.bool)
            visImage = overlay_image_with_tr_mask(imgL, imgTR)

            # Сохранение
            plt.imsave(f"{outdirpath}/{imgTRname}", visImage)

#            # Отображение
#            plt.figure()
#            plt.imshow(visImage)
#            plt.title(f"traversable region ({imgLname})")
#            plt.show()
        else:
            print(f"WARN: file {imgLname} or {imgTRname} not exist")
    print("INFO: SUCCESS")
    print("....: execution time: {:.1f}s.".format(time.clock() - start_time))
