# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:30:20 2018

@author: murata
"""

import numpy as np
import csv, os

import readmhd

#path_to_petct = "../../PET-CT_iso3mm/"
#pth_to_petiso = "../../PET-CT_iso3mm/%s/PETiso.mhd" # % patient_id
##path_to_text = "./pet.txt"
#path_to_csv = "./image_info.csv"
#
#image_info_csv = open(path_to_csv, 'w')
#writer = csv.writer(image_info_csv, lineterminator='\n') 
#writer.writerow( ["patient_id", "voxel_size1", "voxel_size2", "voxel_size3", "matrix_size1", "matrix_size2", "matrix_size3"] ) # headder
#image_info_csv.close()
#
#for patient_id in os.listdir(path_to_petct):
#    print(patient_id)
#    volume = readmhd.read(pth_to_petiso % patient_id)
#    image_info_csv = open(path_to_csv, 'a')
#    writer = csv.writer(image_info_csv, lineterminator='\n') 
#    writer.writerow( [patient_id, volume.voxelsize[0], volume.voxelsize[1], volume.voxelsize[2], volume.matrixsize[0], volume.matrixsize[1], volume.matrixsize[2]] )
#    image_info_csv.close()
#
    
def get_patients(matrix_size=[166,166,257],
                 ):
    path_to_petct = "../../PET-CT_iso3mm/"
    pth_to_petiso = "../../PET-CT_iso3mm/%s/PETiso.mhd" # % patient
    
    patients=[]
    for patient in os.listdir(path_to_petct):
        volume = readmhd.read(pth_to_petiso % patient)
#        print(volume.voxelsize)
        if volume.matrixsize==matrix_size:
            patients.append(patient)
    
    return patients

def get_lung(matrix_size=[166,166,257]):
    path_to_petct = "../../PET-CT_iso3mm/"
    pth_to_petiso = "../../PET-CT_iso3mm/%s/PETiso.mhd" # % patient
    pth_to_LungAreaIso = "../../PET-CT_iso3mm/%s/LungAreaIso.mhd" # % patient
    pth_to_WbMaskIso = "../../PET-CT_iso3mm/%s/WbMaskIso.mhd" # % patient
    path_to_lung_size = "./lung_size.csv"
    
    if os.path.exists(path_to_lung_size):
        os.remove(path_to_lung_size)
    lung_size_csv = open(path_to_lung_size, 'w')
    writer = csv.writer(lung_size_csv, lineterminator='\n') 
    writer.writerow( ["patient_id", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"] ) # headder
    lung_size_csv.close()

#    for patient in get_patients(matrix_size=matrix_size):
    BB_z_max, BB_z_min = 0, 1000
    num_patient, num_lai, num_wmi = 0, 0, 0
    for patient in os.listdir(path_to_petct):
        num_patient += 1
        pet = readmhd.read(pth_to_petiso % patient).vol
        if os.path.exists(pth_to_LungAreaIso % patient):
            num_lai += 1
            lungarea = readmhd.read(pth_to_LungAreaIso % patient).vol
        elif os.path.exists(pth_to_WbMaskIso % patient):
            num_wmi += 1
            lungarea = readmhd.read(pth_to_WbMaskIso % patient).vol
            lungarea[lungarea<3]=0
        BB_lungarea = np.argwhere(lungarea)
        (zmin, ymin, xmin), (zmax, ymax, xmax) = BB_lungarea.min(0), BB_lungarea.max(0)+1
        lung_size_csv = open(path_to_lung_size, 'a')
        writer = csv.writer(lung_size_csv, lineterminator='\n') 
        writer.writerow( [patient, xmin, xmax, ymin, ymax, zmin, zmax] ) # headder
        lung_size_csv.close()
        if zmax-zmin > BB_z_max:
            BB_z_max = zmax-zmin
        if zmax-zmin < BB_z_min:
            BB_z_min = zmax-zmin
#        if zmax-zmin > 100:
#            print(zmax-zmin, patient)
    print(BB_z_min, BB_z_max)
    print(num_patient, num_lai, num_wmi)
#        print(z_min, z_max)
#        print(np.sum(lungarea)==np.sum(lungarea[z_min:z_max,:,:]))

#            print(np.unique(lungarea))
#            print(lungarea.shape)
    
    
    
def main():
    get_lung()
    
#    print(len(patients))
    

if __name__ == '__main__':
    main()

