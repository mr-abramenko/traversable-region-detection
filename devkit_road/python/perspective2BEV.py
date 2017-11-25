#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
perspective2BEV.py

Преобразование из "перспективного вида" к "виду сверху" (см. [1]).

[1] Fritsch J., Kuehnl T., Geiger A. A New Performance Measure and Evaluation
    Benchmark for Road Detection Algorithms International Conference on
    Intelligent Transportation Systems (ITSC) / 2013.
"""
import transform2BEV
import argparse
import time
import sys
import os

# =============================================================================
# Скипт
# =============================================================================
if __name__ == "__main__":
    start_time = time.clock()

    # Анализ аргументов командной строки
    parser = argparse.ArgumentParser(prog='python perspective2BEV.py',
                                 description="""Evaluate road estimation
                                 performance in the bird's-eye-view space""",
                                 epilog="Abramenko A.A.")
    parser.add_argument('result_dir',
                        help="path to perspective traversable maps",
                        metavar="RESULT_PERSP")
    parser.add_argument('gt_dir',
                        help="path to perspective ground truth (gt_image_2)",
                        metavar="GT")
    parser.add_argument('calib_dir',
                        help="path to directory containing calib .txt data",
                        metavar="CALIB")
    parser.add_argument('odir',
                        help="path to output directory",
                        metavar="ODIR")
    parser.add_argument('-v',
                        action='version',
                        version='%(prog)s 1.0.0')
    args = parser.parse_args()

    resultpath = os.path.abspath(args.result_dir)
    gtpath = os.path.abspath(args.gt_dir)
    calibpath = os.path.abspath(args.calib_dir)
    outdirpath = os.path.abspath(args.odir)

    if not(os.path.isdir(resultpath)
            and os.path.isdir(gtpath)
            and  os.path.isdir(calibpath)):
        print("INFO: UNSUCCESS")
        print("....: invalid path to input directories")
        sys.exit(1)

    # RESULTS --> BEV convertation
    print("INFO: RESULTS --> BEV convertation...")
    resultfiles = os.path.join(resultpath, '*.png')
    transform2BEV.main(resultfiles, calibpath, outdirpath)

    # GT --> BEV convertation
    print("INFO: GT --> BEV convertation...")
    gtfiles = os.path.join(gtpath, '*.png')
    transform2BEV.main(gtfiles, calibpath, f"{outdirpath}/gt_image_2/")

    print("INFO: SUCCESS")
    print("....: execution time: {:.1f}s.".format(time.clock() - start_time))