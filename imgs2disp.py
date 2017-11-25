#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imgs2disp.py

Строит карту диспаратности для изображений с помощью алгоритма StereoSGBM.
"""
import numpy as np
import argparse
import time
import glob
import sys
import cv2
import os


def compute_disp(sgbm_obj, imgLname, imgLdirpath,
                 imgRname, imgRdirpath, outdirpath):
    """Вычисляет и сохраняет карту диспаритета для стереопары"""

    if ( os.path.exists(f"{imgLdirpath}/{imgLname}")
            and os.path.exists(f"{imgRdirpath}/{imgRname}")):
        imgL = cv2.imread(f"{imgLdirpath}/{imgLname}", cv2.IMREAD_UNCHANGED)
        imgR = cv2.imread(f"{imgRdirpath}/{imgRname}", cv2.IMREAD_UNCHANGED)

        disp = sgbm_obj.compute(imgL, imgR)

        mask = (disp < 0)
        disp = disp.astype(np.uint16)
        disp[mask] = 65535

        cv2.imwrite(f"{outdirpath}/{imgLname}", disp)
        return 0
    else:
        return 1

# =============================================================================
# Скипт вычисления карт диспарантности
# =============================================================================
if __name__ == "__main__":
    start_time = time.clock()

    # Анализ аргументов командной строки
    parser = argparse.ArgumentParser(prog='python imgs2disp.py',
                                 description="Compute disparity map.",
                                 epilog="Abramenko A.A.")
    parser.add_argument('imgl',
                        help="path to the left image(s)",
                        metavar="IMG_L")
    parser.add_argument('imgr',
                        help="path to the right image(s)",
                        metavar="IMG_R")
    parser.add_argument('odir',
                        help="path to output directory",
                        metavar="ODIR")
    parser.add_argument('-v',
                        action='version',
                        version='%(prog)s 1.0.0')
    args = parser.parse_args()

    workdir = os.getcwd()
    imgLpath = os.path.abspath(args.imgl)
    imgRpath = os.path.abspath(args.imgr)
    outdirpath = os.path.abspath(args.odir)

    if not os.path.exists(outdirpath):
        os.makedirs(outdirpath)

    if ( os.path.isfile(imgLpath) and os.path.isfile(imgRpath) ):
        imgLdirpath = os.path.dirname(imgLpath)
        imgRdirpath = os.path.dirname(imgRpath)

        imgLfilenames = [os.path.basename(imgLpath)]
        imgRfilenames = [os.path.basename(imgRpath)]
    elif ( os.path.isdir(imgLpath) and os.path.isdir(imgRpath) ):
        imgLdirpath = imgLpath
        imgRdirpath = imgRpath

        os.chdir(imgLdirpath)
        imgLfilenames = (glob.glob('*.png') +
                         glob.glob('*.jpg') +
                         glob.glob('*.jpeg') )
        imgRfilenames = imgLfilenames
        os.chdir(workdir)
    else:
        print("INFO: UNSUCCESS")
        print("....: invalid path to input files")
        sys.exit(1)

    # Создание объекта SGBM
    window_size = 7
    num_img_channels = 3
    sgbm_obj = cv2.StereoSGBM_create(minDisparity = 0,
                                        numDisparities = 7*16,
                                        blockSize = 5,
                                        P1 = 8*num_img_channels*window_size**2,
                                        P2 = 32*num_img_channels*window_size**2,
                                        disp12MaxDiff = -1,
                                        uniquenessRatio = 5,
                                        preFilterCap = 1,
                                        speckleWindowSize = 200,
                                        speckleRange = 1,
                                        mode = cv2.StereoSGBM_MODE_SGBM_3WAY
                                        )

    # Вычисление карт диспарантности для входных данных
    print("INFO: Disparity maps executing...")
    for imgLname, imgRname in zip(imgLfilenames, imgRfilenames):
        print("....:", imgLname)
        compute_disp(sgbm_obj, imgLname, imgLdirpath,
                     imgRname, imgRdirpath, outdirpath)
    print("INFO: SUCCESS")
    print("....: execution time: {:.1f}s.".format(time.clock() - start_time))

