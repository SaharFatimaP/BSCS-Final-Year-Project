import numpy as np
import math

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
    object_hight_on_sensor = camera_config[f'sensor_{mode}(mm)'] * 1/camera_config[f'sensor_{mode}(pixels)']
    pixel_size_meters = depth * object_hight_on_sensor /camera_config['focal_length']
    return pixel_size_meters

#print(pixel_size_in_meters(0.7))

def length_of_depth_vec(depth_vec, mode=CalculationMode.Height):
    """
        Calculates the actual length of a given depth vector
    """
    depth_vec_len = len(depth_vec)
    total_length = 0
    lst = []
    h_lst =[]

    # can be made efficient
    for i in range(depth_vec_len-1):
        # getting depth of Pixel 1 and Pixel 2
        depth_p1 = depth_vec[i]
        depth_p2 = depth_vec[i+1]

        # getting depth of base of triangle
        base_pixel_depth = max((depth_p1,depth_p2))
       
        
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

