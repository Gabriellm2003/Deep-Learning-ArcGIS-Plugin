import sys
import os
import shutil

path = sys.argv[1]
'Function to remove empty folders'
# if not os.path.isdir(path):
#     return


# remove empty subfolders
folders = os.listdir(path)  
if len(folders):
    for f in folders:
        fullpath = os.path.join(path, f)
        if os.path.isdir(fullpath):
            models = [x for x in os.listdir(fullpath) if '.h5' in x]
            models.sort()
            for m in range(len(models) - 1):
                print(os.path.join(fullpath, models[m]))
                os.remove(os.path.join(fullpath, models[m]))