import gzip
import os
import io

#解压
def un_gz(file_name,target_dir):
    #文件名，目标路径
    f_name = file_name.replace('.gz',"")
    f_name = f_name.split('\\')
    f_name = os.path.join(target_dir,f_name[-1])
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()

data_dir = "D:/fmri/fmri_data/oasis_NC"
target_dir = "D:/fmri/fmri_data/raw_oasis_NC"
if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

for x in os.listdir(data_dir):
    sub_dir = os.path.join(data_dir,x)
    for y in os.listdir(sub_dir):
        _ = os.path.join(sub_dir,y)
        for z in os.listdir(_):
            if 'gz' in z:
                un_gz(os.path.join(_,z),target_dir)
                break
        break



