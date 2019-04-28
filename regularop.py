import os
import re
import shutil
import csv
def transport():
    raw_dir = "F:/fmri/preprocessing data"
    target_dir = "F:/fmri/AD"
    pattern = r'^xbcNGS.*\.nii$'
    pattern = re.compile(pattern)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    for x in os.listdir(raw_dir):
        sub_dir = os.path.join(raw_dir,x)
        for y in os.listdir(sub_dir):
            if pattern.fullmatch(y):
                raw_filename = os.path.join(sub_dir,y)
                filename = x+'.nii'
                target_filename = os.path.join(target_dir,filename)
                shutil.copy(raw_filename,target_filename)
    print("transport success")

def get_subject():
    filename = "C:/Users/Administrator/Desktop/ADNI/jlfMRI_MCIc_DATASET_3_25_2018.csv"
    write_filename = "MCIc.txt"
    file = csv.DictReader(open(filename,'r'))
    fhandle = open(write_filename,'w')
    for x in file:
        fhandle.write(x['Subject']+',')
# transport()
get_subject()