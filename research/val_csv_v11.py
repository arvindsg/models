import csv
import cv2
import os
import numpy as np
import tqdm
from skimage import io
import logging
FOLDER = 'object_detection/projects/damage_v11/images/train/'
CSV_FILE = 'object_detection/projects/damage_v11/data/damage_v11_train_labels.csv'
OUT_FILE = 'object_detection/projects/damage_v11/data/damage_v11_train_labels_validated.csv'
with open(CSV_FILE, 'r') as fid:
    with open(OUT_FILE, 'w') as wid:

        print('Checking file:', CSV_FILE, 'in folder:', FOLDER)
    
        file = csv.reader(fid, delimiter=',')
        writer = csv.writer(wid,delimiter=',')
        first = True
    
        cnt = 0
        error_cnt = 0
        error = False
        for row in tqdm.tqdm(file):
            if error == True:
                error_cnt += 1
                error = False
    
            if first == True:
                first = False
                continue
    
            cnt += 1
    
            json_name,name, width, height, xmin, ymin, xmax, ymax = (row[0],row[1], int(float(row[2])), int(float(row[3])), int(float(row[5])), int(float(row[6])), int(float(row[7])), int(float(row[8])))
            path = os.path.join(FOLDER, name)
            try:
                _ = io.imread(path)
                img = cv2.imread(path)
        
                if type(img) == type(None):
                    error = True
                    print('Could not read image', img,'at path',path)
        
                org_height, org_width = img.shape[:2]
        
                if org_width != width:
                    error = True
                    print('Width mismatch for image: ', name, width, '!=', org_width)
        
                if org_height != height:
                    error = True
                    print('Height mismatch for image: ', name, height, '!=', org_height)
        
                if xmin > org_width:
                    error = True
                    print('XMIN > org_width for file', name)
        
                if xmax > org_width:
                    error = True
                    print('XMAX > org_width for file', name)
        
                if ymin > org_height:
                    error = True
                    print('YMIN > org_height for file', name)
        
                if ymax > org_height:
                    error = True
                    print('YMAX > org_height for file', name)
        
                if xmin == xmax:
                    error = True
                    print('Delete that box', name)
        
                if ymin == ymax:
                    error = True
                    print('Delete that box', name)
        
                if (xmax * ymax) <= 0:
                    error = True
                    print('Area of a bounding box cannot be negative', name)
        
                if (org_height - ymax) < 0:
                    error = True
                    print('Bigger than the original height', name)
        
                if (org_width - xmax) < 0:
                    error = True
                    print('Bigger than the original width', name)
        
                if xmin > xmax:
                    error = True
                    print("xmin bigger than the xmax", name)
        
                if ymin > ymax:
                    error = True
                    print("ymin bigger than the ymax", name)
        
                if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                    error = True
                    print("Negative values are not allowed", name)
            except KeyboardInterrupt:
                logging.exception("Keyboard interrupt")
                break
            except:
                error=True
                logging.exception("Failed in processing imagge{}".format(path))
        
            if error == True:
                print('Error for file: %s' % name)
                print()
            else:
                writer.writerow(row)

    print('Checked %d files and realized %d errors' % (cnt, error_cnt))
