import numpy
import cv2
import os
import requests
from PIL import Image as PillowImage
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
        self.H = 1080
        self.W = 720
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_annotation_by_filename(self, filaname_coco_annnotation, filename_image):

        coco = COCO(filaname_coco_annnotation)
        keys = numpy.array([k for k in coco.imgToAnns.keys()])
        filenames = numpy.array([coco.imgs[coco.imgToAnns[key][0]['image_id']]['file_name'] for key in keys])
        if filename_image not in filenames:
            return None

        frame_annotations = coco.imgToAnns[keys[filenames == filename_image][0]]

        return frame_annotations

# ----------------------------------------------------------------------------------------------------------------------
    def download_image_by_filename(self, filaname_coco_annnotation, filename_image):
        coco = COCO(filaname_coco_annnotation)
        filenames = numpy.array([im['file_name'] for im in coco.dataset['images']])
        URLs = numpy.array([im['coco_url'] for im in coco.dataset['images']])
        if filename_image not in filenames:
            return None
        URL = URLs[filenames==filename_image][0]
        image = None

        try:
            response = requests.get(URL, stream=True, timeout=2, allow_redirects=False)
        except:
            return image

        if not response.ok:
            return image

        try:
            image = cv2.cvtColor(numpy.array(PillowImage.open(response.raw)), cv2.COLOR_RGB2BGR)
        except:
            return image

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def draw_bboxes(self, frame_annotations, image):

        category_IDs = [annotation['category_id'] for annotation in frame_annotations]
        boxes = [[a['bbox'][1], a['bbox'][0], a['bbox'][1] + a['bbox'][3], a['bbox'][0] + a['bbox'][2]] for a in frame_annotations]
        if image is None:
            image = numpy.full((self.H, self.W, 3), self.col_bg, dtype=numpy.uint8)

        for box, category_id in zip(boxes, category_IDs):
            # image = tools_draw_numpy.draw_rect(image, box[1], box[0], box[3], box[2],color=colors[category_id],label=category_names[category_id])
            image = tools_draw_numpy.draw_rect(image, box[1], box[0], box[3], box[2], color=(200, 0, 0), alpha_transp=0.8)

        return image

# ----------------------------------------------------------------------------------------------------------------------
    def draw_contours(self, frame_annotations, image):
        points = numpy.array([a['segmentation'] for a in frame_annotations]).reshape((-1, 2)).astype(float)
        if image is None:
            image = numpy.full((self.H, self.W, 3), self.col_bg, dtype=numpy.uint8)

        image = tools_draw_numpy.draw_contours(image,points,color=(0,255,255),w=1,transperency=0.9)

        return image
# ----------------------------------------------------------------------------------------------------------------------
    def process_folder(self, filaname_coco_annnotation, folder_images):

        coco = COCO(filaname_coco_annnotation)
        # colors = tools_draw_numpy.get_colors(1 + len(coco.cats))
        # category_names = ['background']+[coco.cats[key]['name'] for key in coco.cats]

        for key in coco.imgToAnns.keys():
            frame_annotations = coco.imgToAnns[key]
            image_id = frame_annotations[0]['image_id']
            filename_image = coco.imgs[image_id]['file_name']
            if os.path.isfile(folder_images + filename_image):
                image = cv2.imread(folder_images + filename_image)
            else:
                image = numpy.full((self.H,self.W,3),self.col_bg,dtype=numpy.uint8)

            image = self.draw_bboxes(frame_annotations, image)

            cv2.imwrite(self.folder_out + filename_image, image)
        return
# ----------------------------------------------------------------------------------------------------------------------

