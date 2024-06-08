import os
import sys
import re
import random

path = sys.argv[-1]
# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)
# keys = list(range(725, 875))
name_dict = dict([(key, []) for key in range(650, 890)])

for filename in fileList:
    if filename == 'before':
        continue
    list_ = re.split(r'[\./-]', filename)
    auc_item = [i for i in list_ if 'auc' in i]
    auc = int(auc_item[0][:3])
    if auc in name_dict:
        name_dict[auc] += [filename]

for auc, value in name_dict.items():
    if len(value) == 0:
        continue
    else:
        number = random.randint(0, len(value)-1)
        for i, filename in enumerate(value):
            oldname = path + os.sep + filename
            if number == i:
                path_list = re.split(r'[\./-]', filename)
                path_list.pop(-1)
                path_list.insert(0, path_list.pop(2))
                new_filename = '-'.join(path_list) + '.pt'
                newname = path + os.sep + new_filename
                # newname = oldname
            else:
                newname = path + os.sep + 'no.pt'
            os.rename(oldname, newname)
