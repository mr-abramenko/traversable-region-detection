#!/bin/bash

# Получение карт диспаритета
python imgs2disp.py ./data/data_road/training/image_2 ./data/data_road_right/training/image_3 ./results/disp_sgbm

# Детекция доступных для движения регионов
python find_traversable.py ./results/disp_sgbm ./results/sgbm_tr_regs_persp

# Объединение изображения и региона 
python overlay_img_with_tr_regs.py ./data/data_road/training/image_2 ./results/sgbm_tr_regs_persp ./results/visimg

# Вычисисление харакреристик для оценки
cd ./devkit_road/python/

python perspective2BEV.py ../../results/sgbm_tr_regs_persp ../../data/data_road/training/gt_image_2 ../../data/data_road/training/calib ../../results/sgbm_tr_regs_bev

python eval_performance.py ../../results/sgbm_tr_regs_bev

cd ../../

# Все результаты сохраняются в папку ./result/
