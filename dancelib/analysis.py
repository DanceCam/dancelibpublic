import sewpy
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def euclidean_distance(coord1, coord2):
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5

def find_matching_coordinates(ground_truth, inferred, tolerance=2):
    matching_coords = []
    for coord in inferred:
        for truth_coord in ground_truth:
            if euclidean_distance(coord, truth_coord) <= tolerance:
                matching_coords.append(coord)
                break
    return matching_coords

def find_false_positives(ground_truth, inferred, tolerance=2):
    false_positives = []
    for coord in inferred:
        match_found = False
        for truth_coord in ground_truth:
            if euclidean_distance(coord, truth_coord) <= tolerance:
                match_found = True
                break
        if not match_found:
            false_positives.append(coord)
    return false_positives

def calculate_precision_recall_fmeasure(ground_truth, inferred, tolerance=2):
    matching_coords = find_matching_coordinates(ground_truth, inferred, tolerance)
    false_positives = find_false_positives(ground_truth, inferred, tolerance)
    
    true_positives = len(matching_coords)
    false_negatives = len(ground_truth) - true_positives
    
    precision = true_positives / (true_positives + len(false_positives))
    recall = true_positives / (true_positives + false_negatives)
    
    # To avoid division by zero in case both precision and recall are zero
    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f_measure

def get_recall_prec_fmeas_perthresh(image_cmp, image_gt, min_thresh=0.1, max_thresh=40, num_thresh=50, verbose=False, sexpath='/opt/conda/bin/sex'):
    
    apertures=(5, 10, 15)
    apertures_str = ', '.join(str(a) for a in apertures)
    
    # Get x, y coordinates from ground truth image
    hdu = fits.PrimaryHDU(image_gt)
    hdu.writeto('256x256.ground_truth_test.fits', overwrite=True)
    sew = sewpy.SEW(sexpath=sexpath,
                    params=["XWIN_IMAGE", "YWIN_IMAGE", "FLUX_ISO", "MAG_APER(3)", "MAGERR_APER(3)", "FLUX_RADIUS",
                            "FLUX_ISO", "MAG_ISO", "MAG_AUTO", "FLAGS"],
                    config={"DETECT_MINAREA": 3, "DETECT_THRESH": 56, "PHOT_APERTURES": apertures_str},
                    loglevel="CRITICAL")
    out_sew_gt = sew("256x256.ground_truth_test.fits")
    gt_table = out_sew_gt['table']
    x_gt, y_gt = gt_table['XWIN_IMAGE'].data, gt_table['YWIN_IMAGE'].data
    xy_gt = list(zip(*(x_gt, y_gt)))
    sew._clean_workdir()
    
    # Write comparison image to fits file
    hdu = fits.PrimaryHDU(image_cmp)
    hdu.writeto('256x256.thresh_test.fits', overwrite=True)
    
    # Initialize parameters
    thresh_list = []
    precision_list = []
    recall_list = []
    fmeasure_list = []
    
    # Iterate over thresholds, acquire (x, y) coordinates, and determine precision, recall, f-measure for detections, save thresh with best chosen metric
    for thresh in tqdm(np.linspace(min_thresh,max_thresh,num_thresh), desc="Threshold Selection Progress"):
        sew = sewpy.SEW(sexpath=sexpath,
                    params=["XWIN_IMAGE", "YWIN_IMAGE", "FLUX_ISO", "MAG_APER(3)", "MAGERR_APER(3)", "FLUX_RADIUS",
                            "FLUX_ISO", "MAG_ISO", "MAG_AUTO", "FLAGS"],
                    config={"DETECT_MINAREA": 3, "DETECT_THRESH": thresh, "PHOT_APERTURES": apertures_str, "DEBLEND_MINCONT": 0.000001, "CLEAN": "N"},
                    loglevel="CRITICAL")
        out_sew = sew("256x256.thresh_test.fits")
        table = out_sew['table']
        #table = table[table['FLAGS'] <= 1]
        sew._clean_workdir()

        x, y = table['XWIN_IMAGE'].data, table['YWIN_IMAGE'].data
        xy = list(zip(*(x, y)))
        precision, recall, f_measure = calculate_precision_recall_fmeasure(xy_gt, xy)
        
        thresh_list.append(thresh)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(f_measure)
        
    return thresh_list, precision_list, recall_list, fmeasure_list


def print_metrics_table(thresholds, recalls, precisions, fmeasures, column_spacing=12, title=''):
    """
    Prints a formatted table of detection metrics including thresholds, recalls,
    precisions, and F-measures.

    Parameters:
    - thresholds: list of float
      List of threshold values.
    - recalls: list of float
      List of recall values.
    - precisions: list of float
      List of precision values.
    - fmeasures: list of float
      List of F-measure values.
    - column_spacing: int
      Spacing between columns in characters.
    """

    # Print the title of the table if provided
    if title:
        print(title)
        print('-' * column_spacing * 4)  # Print a separator line

    # Zip the input lists together for easier iteration
    metrics_list = list(zip(thresholds, recalls, precisions, fmeasures))

    # Print the first line of the titles
    print(f"{'Detection':<{column_spacing}}{'':<{column_spacing}}"
          f"{'':<{column_spacing}}{'':<{column_spacing}}")

    # Print the second line of the titles
    print(f"{'Threshold':<{column_spacing}}{'Recall':<{column_spacing}}"
          f"{'Precision':<{column_spacing}}{'F-Score':<{column_spacing}}")

    # Print the values under each corresponding title
    for values in metrics_list:
        print(f'{values[0]:<{column_spacing}.2f}{values[1]*100:<{column_spacing}.2f}'
              f'{values[2]*100:<{column_spacing}.2f}{values[3]*100:<{column_spacing}.2f}')