import numpy as np


def dice_coefficient(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Computes the Dice coefficient.

    Returns:
        Dice coefficient (0 = no overlap, 1 = perfect overlap)
    """
    assert prediction.shape == target.shape

    prediction_bool = prediction.astype(np.bool)
    target_bool = target.astype(np.bool)

    if not np.any(prediction_bool) and not np.any(target_bool):
        # Matching empty sets is valid so return 1
        return 1.0

    intersection = np.logical_and(prediction_bool, target_bool)

    if not np.any(intersection):
        # Avoid divide by zero
        return 0.0

    return 2.0 * intersection.sum() / (prediction_bool.sum() + target_bool.sum())


def masked_dice_coefficient(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Computes the Dice coefficient without taking under consideration missing gt

    Returns:
        Dice coefficient (0 = no overlap, 1 = perfect overlap)
    """
    assert prediction.shape == target.shape


    mask = target == -1

    prediction = prediction[~mask]
    target = target[~mask]

    prediction_bool = prediction.astype(np.bool)
    target_bool = target.astype(np.bool)

    if not np.any(prediction_bool) and not np.any(target_bool):
        # Matching empty sets is valid so return 1
        return 1.0

    intersection = np.logical_and(prediction_bool, target_bool)

    if not np.any(intersection):
        # Avoid divide by zero
        return 0.0

    return 2.0 * intersection.sum() / (prediction_bool.sum() + target_bool.sum())


# if __name__ == '__main__':
#
#     ##Test masking loss
#     import numpy as np
#     target = np.zeros((3,10,10))
#     target[1] = np.ones((10, 10))
#     target[2] = np.zeros((10, 10)) - 1
#     pred =np.zeros((3,10,10))
#     pred[1] = np.ones((10, 10)) - 0.2
#     pred[2] =  np.ones((10, 10)) - 0.2
#
#
#
#     d = masked_dice_coefficient(pred, target)
#
#     print("Dice", d)
#
#     d = dice_coefficient(pred, target)
#
#     print("Dice", d)