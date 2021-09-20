# First import the library
import pyrealsense2 as rs
from PIL import Image
import numpy as np
import cv2
import glob


def collect_data(image_save_path, depth_save_path, from_image_no):

    image_no = int(from_image_no)
    # Create a context object.
    # This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # post processing
    # temporial filter
    temporal_filter = rs.temporal_filter(
        smooth_alpha=0.4, smooth_delta=20, persistence_control=2)
    # threshold filter
    threshold_filter = rs.threshold_filter(
        min_dist=0.15000000596046448, max_dist=16.0)
    # Colorize
    colorizer = rs.colorizer(0)
    # decimation filter
    decimation_filter = rs.decimation_filter(3)
    # Spatial Filter
    spatial_filter = rs.spatial_filter()

    # Hole filling filter
    hole_filling_filter = rs.hole_filling_filter(1)

    while True:
        # for preview of images
        while(True):
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            color = np.asanyarray(color.get_data())
            cv2.imshow("Image", color)

            align = rs.align(rs.stream.color)
            frames = align.process(frames)

            aligned_depth_frame = frames.get_depth_frame()
            # filter
            filtered = threshold_filter.process(
                aligned_depth_frame)  # aligned_depth_frame
            filtered = hole_filling_filter.process(filtered)
            filtered = colorizer.process(filtered)

            depth = filtered.get_data()
            depth = np.asanyarray(depth, dtype=np.float) * depth_scale
            cv2.imshow("Depth", depth)

            k = cv2.waitKey(30)
            if k == 27:  # if ESC is pressed, close the program
                exit()
            elif k == 32:
                break
            else:
                continue

        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...)
        # on a device will return stable values until wait_for_frames(...) is called
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

        color = np.asanyarray(color.get_data())
        img_color = Image.fromarray(color)
        img_color.save(f'{image_save_path}/{image_no}.jpg')

        # Create alignment primitive with color as its target stream:

        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Update color and depth frames:
        aligned_depth_frame = frames.get_depth_frame()

        # filter
        #filtered = temporal_filter.process(aligned_depth_frame)
        filtered = threshold_filter.process(
            aligned_depth_frame)  # aligned_depth_frame
        #filtered = decimation_filter.process(aligned_depth_frame)
        #filtered = spatial_filter.process(filtered)
        filtered = hole_filling_filter.process(filtered)

        depth = filtered.get_data()
        depth = np.asanyarray(depth, dtype='float') * depth_scale

        # saving the depth data
        np.save(f'{depth_save_path}/{image_no}.npy', depth)
        image_no += 1
        print(f"Image #{image_no} saved")


if __name__ == "__main__":
    # input start image
    image_no = input("Please input starting image number: ")
    image_save_path = './Dataset/Experiments'
    depth_save_path = './Dataset/Experiments'

    collect_data(image_save_path, depth_save_path, image_no)
