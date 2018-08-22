# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:30:20 2018

@author: murata
"""

import numpy as np
import csv, os

import readmhd

path_to_petct = "../../PET-CT_iso3mm/"
pth_to_petiso = "../../PET-CT_iso3mm/%s/PETiso.mhd" # % patient_id
#path_to_text = "./pet.txt"
path_to_csv = "./image_info.csv"

image_info_csv = open(path_to_csv, 'w')
writer = csv.writer(image_info_csv, lineterminator='\n') 
writer.writerow( ["patient_id", "voxel_size1", "voxel_size2", "voxel_size3", "matrix_size1", "matrix_size2", "matrix_size3"] ) # headder
image_info_csv.close()

for patient_id in os.listdir(path_to_petct):
    print(patient_id)
    volume = readmhd.read(pth_to_petiso % patient_id)
    image_info_csv = open(path_to_csv, 'a')
    writer = csv.writer(image_info_csv, lineterminator='\n') 
    writer.writerow( [patient_id, volume.voxelsize[0], volume.voxelsize[1], volume.voxelsize[2], volume.matrixsize[0], volume.matrixsize[1], volume.matrixsize[2]])
    image_info_csv.close()