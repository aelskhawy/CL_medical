import logging
from pathlib import Path
from typing import Tuple, List

# import SimpleITK as sitk
# from edipy.dicom.metadata import SeriesTag
# from edipy.sitk import loading


import numpy as np

import pydicom

import os

logger = logging.getLogger(__name__)
watermark = 7000

def company_formatted_id(patient_id: int) -> str:
    return str(patient_id).zfill(5)


def load_series(dicom_folder: Path, series_number: int, use_calculated_padding: bool = False,
                load_metadata_only: bool = False): # -> sitk.Image:
    metadata_query = {SeriesTag.SeriesNumber: series_number}
    return loading.load_dicom_from_query(root=dicom_folder, metadata_query=metadata_query,
                                         use_calculated_padding=use_calculated_padding,
                                         load_metadata_only=load_metadata_only)


def load_image_volume(path: Path, watermark: int =watermark) -> Tuple[np.ndarray, List]:
    """
    Load image volume that is not in edidicom and obtain spacing
    :param path: Dicom path
    :return: image volume and spacing
    """
    # load image volume
    scan = load_scan(str(path))
    image_volume = get_pixels_hu(scan)
    image_spacing = get_spacing(scan)
    # make padding values consistent between volumes, at -1024
    image_volume = np.clip(image_volume, a_min=-1024, a_max=np.max(image_volume))
    image_volume[image_volume == watermark] = -1024  # removing white square marking

    return image_volume, image_spacing


def get_pixels_hu(scans: List) -> np.ndarray:
    """
    Convert scan object to volume containing hu values.
    :param scans: scan object
    :return: volume containing hu values
    """

    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope


    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def get_spacing(scan: list) -> list:
    """
    Given a scan object, return the spacing of the scan.
    :param scan: scan object
    :return: spacing of the scan
    """
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + [scan[0].PixelSpacing[0]] + [scan[0].PixelSpacing[1]]))
    spacing = list(spacing)

    return spacing


def load_scan(path: str)-> List:
    """
    Load in slices.
    :param path: path containing DICOM slices
    :return: slices
    """
    slices = [pydicom.read_file(path + '/' + s, force=True) for s in os.listdir(path) if str(s).endswith('.dcm')]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


