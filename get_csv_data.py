import csv
# filename = 'C:/Users/rong/Documents/Tencent Files/452367730\FileRecv/pandan_1_20_2019_5_28_6_OASIS_table.csv'
filename = 'select_fmri.csv'
# cli_filename = ''
csv_file = csv.reader(open(filename,'r'))
# clinic_file = csv.reader(open(cli_filename,'r'))

with open('NC_fmri.csv','w',newline='') as out: #不加newline会有空行
    csv_write = csv.writer(out,dialect='excel')
    for i ,MR in enumerate(csv_file):
        if i and 'normal' in MR[12]:
            csv_write.writerow(MR)
#     s = set()
#     for i,MR in enumerate(csv_file):
#         if i == 0:
#             csv_write.writerow(MR)
#         else:
#             if MR[2] not in s and 'bold' in MR[5]:
#                 s.add(MR[2])
#                 csv_write.writerow(MR)
    print("write over")