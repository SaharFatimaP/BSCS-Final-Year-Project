# First import the library
import pyrealsense2 as rs
from PIL import Image 
import numpy as np
import cv2


image_no = 100000

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()
print(config)
exit()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()



while True:

    # for preview of images
    while(True):

        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        color = np.asanyarray(color.get_data())

        cv2.imshow("Image", color)

        k = cv2.waitKey(30)
        if k == 27: # if ESC is pressed, close the program
            exit()
        elif k == 32:
            break
        else:
            continue

    # This call waits until a new coherent set of frames is available on a device
    # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()

    color = np.asanyarray(color.get_data())
    img_color = Image.fromarray(color)
    img_color.save('C:\\users\\ateeb\\desktop\\depth_dataset\\{}.jpg'.format(image_no))
    
    # Create alignment primitive with color as its target stream:
    
    align = rs.align(rs.stream.color)
    frames = align.process(frames)

    # Update color and depth frames:
    aligned_depth_frame = frames.get_depth_frame()
    depth = aligned_depth_frame.get_data()
    depth = np.asanyarray(depth , dtype = 'float') * depth_scale

    #saving the depth data
    np.save('C:\\users\\ateeb\\desktop\\depth_\\'+str(image_no) + '.npy' , depth)

    image_no += 1
    print(str(image_no)+'.jpg')