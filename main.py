import numpy
# ----------------------------------------------------------------------------------------------------------------------
import utils_markup_coco
import utils_kitti
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def ex_01_coco_local_folder():
    filename_coco_annnotation = './data/ex_01_coco_local/coco.json'
    folder_images = './data/ex_01_coco_local/'
    M = utils_markup_coco.Markuper(filename_coco_annnotation,folder_out)
    M.process_local_folder(folder_images)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_02_coco_remote():
    filaname_coco_annnotation = './data/ex_02_coco_cloud/person_keypoints_val2017.json'
    M = utils_markup_coco.Markuper(filaname_coco_annnotation,folder_out)
    M.process_remote_storage()
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_03_kitti():
    mat_proj =numpy.array([[7.215377e+02,0.000000e+00, 6.095593e+02, 4.485728e+01],[0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],[0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])
    point_van_xy = (631, 166)
    folder_labels = 'data/ex_03_kitti/2011_09_26_drive_0009_sync/label_02/'
    folder_images = 'data/ex_03_kitti/2011_09_26_drive_0009_sync/image_02/data/'

    M = utils_kitti.Markuper_Kitti(folder_out)
    M.draw_boxes(folder_out, folder_images, folder_labels, mat_proj, point_van_xy)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_04_nuscene():
    mat_proj = numpy.array([[633.0, 0.0, 400.0, 0.0], [0.0, 633.0, 224.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    point_van_xy = (448, 237)
    folder_labels = 'data/ex_04_NuScene/label_02/'
    folder_images = 'data/ex_04_NuScene/image_02/data/'

    M = utils_kitti.Markuper_Kitti(folder_out)
    M.draw_boxes(folder_out, folder_images, folder_labels, mat_proj, point_van_xy)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #ex_01_coco_local_folder()
    ex_02_coco_remote()
