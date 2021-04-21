import glob
import numpy as np
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut


def isaxial(IOP):
    """
    Check whether an instance is axial
    Parameters
    ----------
    IOP: {list, numpy.ndarray}
        Image Orientation Patient

    Returns: bool
        Axial result
    -------

    """
    IOP = [round(float(s)) for s in IOP]
    if IOP == [1, 0, 0, 0, 1, 0]:
        return True
    else:
        return False


def dcm2image(dcm_path):
    dcm = dcmread(dcm_path)
    array = apply_voi_lut(dcm.pixel_array, dcm)
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        array = np.amax(array) - array

    array = array - np.min(array)
    array = array / np.max(array)
    array = array * 255
    array = array.astype(np.uint8)

    image = np.stack([array] * 3, axis=0)
    image = np.transpose(image, (1, 2, 0))

    return image


def study2series_dict(study_path):
    series_dict = {}

    dcm_paths = glob.glob("{}/*.dcm".format(study_path)) + glob.glob("{}/*.dicom".format(study_path))
    for dcm_path in dcm_paths:
        dcm = dcmread(dcm_path)
        series_uid = dcm.SeriesInstanceUID
        try:
            if series_uid not in series_dict:
                series_dict[series_uid] = [[dcm_path, int(dcm.InstanceNumber), isaxial(dcm.ImageOrientationPatient)]]
            else:
                series_dict[series_uid].append(
                    [dcm_path, int(dcm.InstanceNumber), isaxial(dcm.ImageOrientationPatient)])
        except:
            pass

    for series_uid in list(series_dict.keys()):
        series_instance_planes = [instance[2] for instance in series_dict[series_uid]]
        if (sum(series_instance_planes) / len(series_instance_planes)) < 0.8:
            del series_dict[series_uid]

    for series_uid in list(series_dict.keys()):
        s = sorted(series_dict[series_uid], key=lambda instance: instance[1])
        image = dcm2image(s[len(s) // 2][0])
        series_dict[series_uid] = image

    return series_dict
