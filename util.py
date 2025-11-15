import cv2
import numpy as np

def rotate_image(img, angle, scale=1.0):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, scale)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

def rotate_image_crop(img, angle, scale=1.0):
    rows, cols, _ = img.shape
    center = (cols // 2, rows // 2)
    # Get the 2D rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (cols, rows))

def rotate_point(point, angle_degrees):
    """
    Rotates a 2D point counter-clockwise around the origin (0,0).

    Args:
        point (tuple or list): The (x, y) coordinates of the point.
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The new (x', y') coordinates.
    """
    # 1. Convert angle to radians
    angle_radians = np.radians(angle_degrees)
    
    # 2. Define point vector
    P = np.array(point)
    
    # 3. Create the 2x2 Rotation Matrix
    c, s = np.cos(angle_radians), np.sin(angle_radians)
    R = np.array([
        [c, -s],
        [s,  c]
    ])
    
    # 4. Perform matrix multiplication: P' = R @ P
    P_rotated = R @ P
    
    return P_rotated