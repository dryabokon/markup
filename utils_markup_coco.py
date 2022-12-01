import cv2
import os
from pycocotools.coco import COCO
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
class Markuper:
    def __init__(self, folder_out):
        if not os.path.isdir(folder_out):
            os.mkdir(folder_out)

        self.folder_out=folder_out
        self.col_bg = (32,32,32)

        return

# ----------------------------------------------------------------------------------------------------------------------
    def draw_bboxes(self, filaname_coco_annnotation, folder_images):

        coco = COCO(filaname_coco_annnotation)
        colors = tools_draw_numpy.get_colors(1 + len(coco.cats))
        category_names = ['background']+[coco.cats[key]['name'] for key in coco.cats]

        for key in coco.imgToAnns.keys():

            annotations = coco.imgToAnns[key]
            image_id = annotations[0]['image_id']
            filename = coco.imgs[image_id]['file_name']
            if not os.path.isfile(folder_images + filename):
                continue

            category_IDs = [annotation['category_id'] for annotation in annotations]
            boxes = [[a['bbox'][1], a['bbox'][0], a['bbox'][1] + a['bbox'][3], a['bbox'][0] + a['bbox'][2]] for a in annotations]
            image = cv2.imread(folder_images + filename)
            image[:,:]=self.col_bg

            for box,category_id in zip(boxes,category_IDs):
                #image = tools_draw_numpy.draw_rect(image, box[1], box[0], box[3], box[2],color=colors[category_id],label=category_names[category_id])
                image = tools_draw_numpy.draw_rect(image, box[1], box[0], box[3], box[2],color=colors[category_id],alpha_transp=0)

            cv2.imwrite(self.folder_out + filename, image)
        return
# ----------------------------------------------------------------------------------------------------------------------
