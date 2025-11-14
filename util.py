import numpy as np


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