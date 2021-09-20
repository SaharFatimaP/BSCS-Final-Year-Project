import numpy as np
import math

# https://www.ovt.com/download/sensorpdf/153/OmniVision_OV2740.pdf


class CalculationMode:
    Height = "height"
    Width = "width"


camera_config = {
    'sensor_height(mm)': 1.406,
    'sensor_width(mm)': 2.5,
    'sensor_height(pixels)': 720,
    'sensor_width(pixels)': 1280,
    'focal_length': 1.88,
}


def pixel_size_in_meters(depth, camera_config=camera_config, mode=CalculationMode.Height):
    """
        formulas are taken from the following website:
        https://www.scantips.com/lights/subjectdistance.html
    """
    object_hight_on_sensor = camera_config[f'sensor_{mode}(mm)'] * \
        1/camera_config[f'sensor_{mode}(pixels)']
    pixel_size_meters = depth * object_hight_on_sensor / camera_config['focal_length']
    return pixel_size_meters

# print(pixel_size_in_meters(0.7))


def length_of_depth_vec(depth_vec, mode=CalculationMode.Height):
    """
        Calculates the actual length of a given depth vector
    """
    depth_vec_len = len(depth_vec)
    total_length = 0
    lst = []
    h_lst = []

    # can be made efficient
    for i in range(depth_vec_len-1):
        # getting depth of Pixel 1 and Pixel 2
        depth_p1 = depth_vec[i]
        depth_p2 = depth_vec[i+1]

        # getting depth of base of triangle
        base_pixel_depth = max((depth_p1, depth_p2))

        base = pixel_size_in_meters(base_pixel_depth, camera_config, mode)
        lst.append(base)
        #print(depth_vec[i], depth_vec[i+1],base_pixel_depth,base)
        # getting height of the triangle
        height = abs(depth_p1-depth_p2)
        h_lst.append(height)
        # pythagorean theorem

        hypotenuse = math.sqrt((height**2)+(base**2))
        total_length += hypotenuse

    # last pixel is not incorporated
    last_depth = depth_vec[-1]
    base = pixel_size_in_meters(last_depth, camera_config, mode)
    total_length += base
    return total_length


def body_length_depth_vec(p0, p1, d):

    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    NO_OF_SAMPLES = int(((x1 - x0)**2 + (y1 - y0)**2)**0.5)
    print('NO_OF_SAMPLES', NO_OF_SAMPLES)

    dx = x1 - x0
    dy = y1 - y0

    # calculating 'm' that is: slope
    m = dy/dx

    # calculating 'b' that is: y-intercept.
    b = y1 - m*x1

    # interval on the x-axis
    delta_x = dx / NO_OF_SAMPLES
    x_axis = [x0 + delta_x*i for i in range(1, NO_OF_SAMPLES+1)]

    depth_vec = []
    for x in x_axis:
        # evaluating equation at: x
        y = m*x + b
        # appending the selected point in the depth_vec:
        depth_vec.append(d[int(y), int(x)])

    return depth_vec
