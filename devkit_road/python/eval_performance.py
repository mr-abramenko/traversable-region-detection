#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_performance.py

Оценка результата работы алгоритма согласно методике [1].

[1] Fritsch J., Kuehnl T., Geiger A. A New Performance Measure and Evaluation
    Benchmark for Road Detection Algorithms International Conference on
    Intelligent Transportation Systems (ITSC) / 2013.
"""
import evaluateRoad
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
    parser = argparse.ArgumentParser(prog='python eval_performance.py',
                                 description="""Evaluate road estimation
                                 performance in the bird's-eye-view space""",
                                 epilog="Abramenko A.A.")
    parser.add_argument('result_dir',
                        help="""path to BEV traversable maps
                                (has to contain gt_image_2)""",
                        metavar="RESULT_BEV")
    parser.add_argument('-v',
                        action='version',
                        version='%(prog)s 1.0.0')
    args = parser.parse_args()

    resultpath = os.path.abspath(args.result_dir)

    if not (os.path.isdir(resultpath)
            and os.path.isdir(f"{resultpath}/gt_image_2/")):
        print("INFO: UNSUCCESS")
        print("....: invalid path to input directory")
        print("....: (check that  traversable_maps contains gt_image_2)")
        sys.exit(1)

    # Вычисление характеристик для оценки результата
    evaluateRoad.main(resultpath, resultpath)

    print("INFO: SUCCESS")
    print("....: execution time: {:.1f}s.".format(time.clock() - start_time))