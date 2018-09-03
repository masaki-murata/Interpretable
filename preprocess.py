# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:30:20 2018

@author: murata
"""

import numpy as np
import csv, os

import readmhd

   
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


def lung_center_size(patient="",
                     center_size="center"):
    path_to_LungAreaIso = "../../PET-CT_iso3mm/%s/LungAreaIso.mhd" # % patient
    path_to_WbMaskIso = "../../PET-CT_iso3mm/%s/WbMaskIso.mhd" # % patient

    if os.path.exists(path_to_LungAreaIso % patient):
        lungarea = readmhd.read(path_to_LungAreaIso % patient).vol
    elif os.path.exists(path_to_WbMaskIso % patient):
        lungarea = readmhd.read(path_to_WbMaskIso % patient).vol
        lungarea[lungarea<3]=0
    BB_lungarea = np.argwhere(lungarea)
    (zmin, ymin, xmin), (zmax, ymax, xmax) = BB_lungarea.min(0), BB_lungarea.max(0)+1
    x_center, y_center, z_center = int((xmax+xmin)/2.0), int((ymax+ymin)/2.0), int((zmax+zmin)/2.0)
    
    if center_size=="center":
        return x_center, y_center, z_center
    if center_size=="size":
        return xmax-xmin, ymax-ymin, zmax-zmin
#    return (xmin, xmax, ymin, ymax, zmin, zmax)
    
    
def get_lung_size():
    path_to_petct = "../../PET-CT_iso3mm/"
    path_to_petiso = "../../PET-CT_iso3mm/%s/PETiso.mhd" # % patient
    path_to_LungAreaIso = "../../PET-CT_iso3mm/%s/LungAreaIso.mhd" # % patient
    path_to_WbMaskIso = "../../PET-CT_iso3mm/%s/WbMaskIso.mhd" # % patient
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
#        pet = readmhd.read(path_to_petiso % patient).vol
        if os.path.exists(path_to_LungAreaIso % patient):
            num_lai += 1
            lungarea = readmhd.read(path_to_LungAreaIso % patient).vol
        elif os.path.exists(path_to_WbMaskIso % patient):
            num_wmi += 1
            lungarea = readmhd.read(path_to_WbMaskIso % patient).vol
            lungarea[lungarea<3]=0
        BB_lungarea = np.argwhere(lungarea)
        (zmin, ymin, xmin), (zmax, ymax, xmax) = BB_lungarea.min(0), BB_lungarea.max(0)+1
        print(patient)
        lung_size_csv = open(path_to_lung_size, 'a')
        writer = csv.writer(lung_size_csv, lineterminator='\n') 
        writer.writerow( [patient, xmin, xmax, ymin, ymax, zmin, zmax] ) 
        lung_size_csv.close()
        num_patient += 1
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
#    return 

    
#def test():
#    path_to_petct = "../../PET-CT_iso3mm/"
#    path_to_test_csv = "./test.csv"
#    
#    if os.path.exists(path_to_test_csv):
#        os.remove(path_to_test_csv)
#
#    test_csv = open(path_to_test_csv, 'w')
#    writer = csv.writer(test_csv, lineterminator='\n') 
#    writer.writerow( ["patient"] ) 
#    test_csv.close()
#    
#    for patient in os.listdir(path_to_petct): 
##        print(patient)
#        test_csv = open(path_to_test_csv, 'a')
#        writer = csv.writer(test_csv, lineterminator='\n') 
#        writer.writerow( [patient] ) 
#        test_csv.close()
#        
    
    
def main():
    get_lung()
    
#    print(len(patients))
    

if __name__ == '__main__':
    main()

