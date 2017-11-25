#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_traversable.py

Оcновываясь на карте диспаритета, находит свободные для движения регионы [1].

[1] Zhu X. et al. Stereo vision based traversable region detection for mobile
    robots using uv-disparity // Control Conference (CCC), 2013 32nd Chinese. -
    IEEE, 2013. - С. 5785-5790.
"""
import numpy as np
import argparse
import time
import glob
import sys
import cv2
import os
from skimage.morphology import remove_small_holes, remove_small_objects
#from matplotlib import pyplot as plt


def read_disp(dispname, dispdirpath):
    """Считывает и возвращает карту диспаритета из файла *.png"""

    disp = cv2.imread(f"{dispdirpath}/{dispname}", cv2.IMREAD_UNCHANGED)

    mask_invalid = (disp == 65535)  # маска для невалидных значений
    disp = disp.astype(np.float32) / 16  # конвертация в формат float32
    disp[mask_invalid] = np.nan  # невалидные значения  обозначены как NaN

#    # Рисуем карту диспаритета
#    plt.figure()
#    plt.imshow(disp)
#    plt.title(f"disparity map ({dispname})")
#    plt.show()

    return disp

def compute_u_disp(disp):
    """Вычисляет u-диспаритет для карты диспарантности"""

    # Нахождение уникальных значений диспаритета
    mask_valid = np.logical_not(np.isnan(disp))
    d_unique = np.unique(disp[mask_valid])
    # Запись биекции значений диспаритета и индекса ввиде словарей
    d2index_dict = dict( zip(d_unique, range(d_unique.size)) )
    index2d_dict = dict( zip(range(d_unique.size), d_unique) )

    # Подсчет кол-ва пикселей вдоль вертикальных направлений с одинаковым
    # диспаритетом и формирование карты u-диспаритета
    m = d_unique.size
    n = disp.shape[1]
    u_disp = np.zeros((m, n))

    for u in range(n):
        unique, counts = np.unique(disp[:,u][mask_valid[:,u]],
                                   return_counts=True)
        u_disp[[d2index_dict[d] for d in unique], u] = counts

    return (u_disp, index2d_dict)

def compute_v_disp(disp):
    """Вычисляет v-диспаритет для карты диспарантности"""

    # Нахождение уникальных значений диспаритета
    mask_valid = np.logical_not(np.isnan(disp))
    d_unique = np.unique(disp[mask_valid])
    # Запись биекции значений диспаритета и индекса ввиде словарей
    d2index_dict = dict( zip(d_unique, range(d_unique.size)) )
    index2d_dict = dict( zip(range(d_unique.size), d_unique) )

    # Подсчет кол-ва пикселей вдоль горизонтальных направлений с одинаковым
    # диспаритетом и формирование карты v-диспаритета
    m = disp.shape[0]
    n = d_unique.size
    v_disp = np.zeros((m, n))

    for v in range(m):
        unique, counts = np.unique(disp[v,:][mask_valid[v,:]],
                                   return_counts=True)
        v_disp[v, [d2index_dict[d] for d in unique]] = counts

    return (v_disp, index2d_dict)

def split_disp(disp, u_disp_threshold=3, morph_disk_radius=9, small_obj_size=500,
               connectivity=1):
    """
    Разделяет карту диспаритета на две карты диспаритета
    (препятствий и не препятствий)
    """
    # Получение u-диспаритета
    (u_disp, index2d_dict) = compute_u_disp(disp)

    # Применение порога и поиск пикселей относящихся к препятствию
    mask_obst = np.zeros_like(disp, dtype=np.uint8)

    u_disp_bin = u_disp > u_disp_threshold
    for (m,u) in np.argwhere(u_disp_bin):
        d = index2d_dict[m]
        mask_obst[np.where(disp[:,u] == d),u] = 255

    # Выполнение морфологической операции замыкания
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_disk_radius,
                                                          morph_disk_radius))
    mask_obst = cv2.morphologyEx(mask_obst, cv2.MORPH_CLOSE, kernel,
                                 iterations=1)

    # Избавление от небольших изолированных регионов
    mask_obst = mask_obst.astype(np.bool)
    mask_obst = remove_small_holes(mask_obst,
                                   min_size=small_obj_size,
                                   connectivity=connectivity)
    mask_obst = remove_small_objects(mask_obst,
                                     min_size=small_obj_size,
                                     connectivity=connectivity)

    # Получение искомых карт диспаритета
    mask_non_obst = np.logical_not(mask_obst)

    invalid = disp.copy()
    invalid[:] = np.nan

    obst_disp = np.where(mask_obst, disp, invalid)
    non_obst_disp = np.where(mask_non_obst, disp, invalid)

#    # Рисуем карту u-диспаритета
#    plt.figure()
#    plt.imshow(u_disp_bin, 'summer')
#    plt.title(f"u-disparity binary map \n(u_disp_threshold={u_disp_threshold})")
#    plt.show()

#    # Рисуем карту диспаритета припятствий
#    plt.figure()
#    plt.imshow(obst_disp)
#    plt.title("obstacle disparity map")
#    plt.show()

#    # Рисуем карту диспаритета не-припятствий
#    plt.figure()
#    plt.imshow(non_obst_disp)
#    plt.title("non-obstacle disparity map")
#    plt.show()

    return (obst_disp, non_obst_disp)

def detect_traversable_regions(filename, outdirpath,
                               non_obst_disp, v_disp_threshold=3,
                               line_width=20, morph_disk_radius=9,
                               small_obj_size=500, connectivity=1):
    """
    Определяет регионы, доступные для движения, и возвращает
    маску для входной карты диспаритета.
    """

    # Получение v-диспаритета
    (v_disp, index2d_dict) = compute_v_disp(non_obst_disp)

    # Нахождение линии кореляции земной поверхности
    # с помощью преобразования Хафа
    v_disp_bin = (v_disp > v_disp_threshold).astype(np.uint8)

    lines = cv2.HoughLines(image=v_disp_bin,
                           rho=1,
                           theta=np.pi/180,
                           threshold=50)
    mask_tr_regions = np.zeros_like(non_obst_disp, dtype=np.uint8)
    if lines is not None:
        rho, theta = lines[0,0,:]  # параметры линии с наибольшим кол. голосов

        for (v,n) in np.argwhere(v_disp_bin != 0):
            condition = n*np.cos(theta) + v*np.sin(theta)
            if ((condition >= rho-line_width/2)
                    and (condition <= rho+line_width/2)):
                d = index2d_dict[n]
                mask_tr_regions[v, np.where(non_obst_disp[v,:] == d)] = 255

        # Избавление от пустот
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_disk_radius,
                                                              morph_disk_radius))
        mask_tr_regions = cv2.morphologyEx(mask_tr_regions, cv2.MORPH_CLOSE,
                                           kernel, iterations=1)

        mask_tr_regions = mask_tr_regions.astype(np.bool)
        mask_tr_regions = remove_small_holes(mask_tr_regions,
                                       min_size=small_obj_size,
                                       connectivity=connectivity)

        # Избавление от маленьких изолированных участков
        mask_tr_regions = remove_small_objects(mask_tr_regions,
                                         min_size=small_obj_size,
                                         connectivity=connectivity)

        mask_tr_regions = mask_tr_regions.astype(np.uint8)*255
        cv2.imwrite(f"{outdirpath}/{filename}", mask_tr_regions)


#        # Рисуем найденную линию корреляции земли линию
#        plt.figure()
#        plt.imshow(v_disp_bin, 'summer')
#        plt.title(f"ground correlation lane ({filename})")
#        x = np.arange(0., v_disp_bin.shape[1], 1)
#        plt.plot(x, ((-np.cos(theta)/np.sin(theta))*x
#                     + (rho+line_width/2)/np.sin(theta)), 'b')
#        plt.plot(x, ((-np.cos(theta)/np.sin(theta))*x
#                     + (rho-line_width/2)/np.sin(theta)), 'b')
#        plt.plot(x, ((-np.cos(theta)/np.sin(theta))*x
#                     + (rho/np.sin(theta))), 'r--')
#        plt.show()

    return mask_tr_regions

# =============================================================================
# Скипт
# =============================================================================
if __name__ == "__main__":
    start_time = time.clock()

    # Анализ аргументов командной строки
    parser = argparse.ArgumentParser(prog='python find_traversable.py',
                                 description="Traversable region detection.",
                                 epilog="Abramenko A.A.")
    parser.add_argument('disp',
                        help="path to input disparity map(s)",
                        metavar="DISP")
    parser.add_argument('odir',
                        help="path to output directory",
                        metavar="ODIR")
    parser.add_argument('-v',
                        action='version',
                        version='%(prog)s 1.0.0')
    args = parser.parse_args()

    workdir = os.getcwd()
    disppath = os.path.abspath(args.disp)
    outdirpath = os.path.abspath(args.odir)

    if not os.path.exists(outdirpath):
        os.makedirs(outdirpath)

    if os.path.isfile(disppath):
        dispdirpath = os.path.dirname(disppath)
        dispfilenames = [os.path.basename(disppath)]
    elif os.path.isdir(disppath):
        dispdirpath = disppath

        os.chdir(dispdirpath)
        dispfilenames = glob.glob('*.png')
        os.chdir(workdir)
    else:
        print("INFO: UNSUCCESS")
        print("....: invalid path to input disparity map(s)")
        sys.exit(1)


    # Реализация алгоритма
    print("INFO: Traversable regions searching...")
    for dispfilename in dispfilenames:

        tags = dispfilename.split('_')
        if len(tags) == 2:
            filename = tags[0] + '_road_' + tags[1]
        else:
            filename = dispfilename
        print("....:", filename)

        disp = read_disp(dispfilename, dispdirpath)

        (obst_disp, non_obst_disp) = split_disp(disp,
                                                u_disp_threshold=3,
                                                morph_disk_radius=9,
                                                small_obj_size=500,
                                                connectivity=1)

        mask_tr_regions = detect_traversable_regions(filename,
                                                     outdirpath,
                                                     non_obst_disp,
                                                     v_disp_threshold=3,
                                                     line_width=20,
                                                     morph_disk_radius=9,
                                                     small_obj_size=500,
                                                     connectivity=1)

    print("INFO: SUCCESS")
    print("....: execution time: {:.1f}s.".format(time.clock() - start_time))

