import os
import cv2
import re
import math
import torch
import numpy as np
from habitat import logger
from habitat_baselines.utils.common import center_crop
import os
import cv2
import numpy as np
from math import ceil
from vlnce_baselines.common.utils import get_camera_orientations
from typing import Dict, List, Optional, Tuple
import tqdm
import imageio

def images_to_video(
    step_dir: str,
    video_name: str = "history",
    fps: int = 2,
    quality: Optional[float] = 5,
    verbose: bool = True,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using 'bitrate' disables
            this parameter.
    """
    # step_dir = os.path.join(output_dir, "history")
    if not os.path.exists(step_dir):
        raise FileNotFoundError(f"Path {step_dir} does not exist! Please check the input path.")
    
    # Get all step_*.png files and sort them by the number after step
    files = [f for f in os.listdir(step_dir) if f.startswith("step") and f.endswith(".png")]
    files = sorted(files, key=lambda x: float(re.search(r"step(\d+\.\d+|\d+)", x).group(1)))
    # Read all images
    images = [cv2.imread(os.path.join(step_dir, f)) for f in files]

    assert 0 <= quality <= 10

    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(step_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    logger.info(f"created: {os.path.join(step_dir, video_name)} Shape:{images[0].shape} fps: {fps}")
    if verbose:
        images_iter = tqdm.tqdm(images)
    else:
        images_iter = images
    for im in images_iter:
        rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_im)
    writer.close()

def pano_mask_side(image):
    """
    Overlay semi-transparent white on the back view of 360° panoramic image.
    
    Args:
        image (ndarray): Input panoramic image.
    Returns:
        ndarray: Panoramic image after masking the back view.
    """
    # pano image
    img_height, img_width = image.shape[:2]
    rect_width = img_width // 8
    
    # Create overlay layer
    output_image = image.copy()
    overlay = image.copy()
    
    # Add semi-transparent white mask on left and right 1/8
    cv2.rectangle(overlay, (0, 0), (rect_width, img_height), (255, 255, 255), -1)
    cv2.rectangle(overlay, (img_width - rect_width, 0), (img_width, img_height), (255, 255, 255), -1)
    
    # Add semi-transparent effect
    cv2.addWeighted(overlay, 0.5, output_image, 0.5, 0, output_image)
    return output_image

def draw_front_mark(image):
    # Add green frame in center
    img_height, img_width = image.shape[:2]
    output_image = image.copy()
    output_image[0:img_height, img_width//8-1:img_width//8+1] = (0, 255, 0)
    output_image[0:img_height, img_width*7//8-1:img_width*7//8+1] = (0, 255, 0)
    output_image[:2,img_width*1//8:img_width*7//8] = (0, 255, 0)
    output_image[-2:,img_width*1//8:img_width*7//8] = (0, 255, 0)
    return output_image

def draw_step_num(image, step_id = 1):
    output_image = image.copy()
    text = f"Step {step_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2
    position = (15, 35)  # Text position
    cv2.putText(output_image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return output_image

def pano_split(image):
    # pano image from left to right is 360 degrees to 0 degrees
    annot_color = (0, 255, 0)
    font_size = 0.6
    font_scale = 2
    img_height, img_width = image.shape[:2]
    image[:, img_width*3//4-1:img_width*3//4+2] = annot_color  # Use white line to separate two images
    # Add annotations below the image
    cv2.putText(image, 'behind', (img_width*7//8 - cv2.getTextSize('behind', cv2.FONT_HERSHEY_SIMPLEX, font_size, font_scale)[0][0]//2, img_height-10), cv2.FONT_HERSHEY_SIMPLEX, font_size, annot_color, font_scale)
    cv2.putText(image, 'front', (img_width*3//8 - cv2.getTextSize('front', cv2.FONT_HERSHEY_SIMPLEX, font_size, font_scale)[0][0]//2, img_height-10), cv2.FONT_HERSHEY_SIMPLEX, font_size, annot_color, font_scale)
    # Label left on the left side of front, right on the right boundary
    cv2.putText(image, 'left', (0, img_height-10), cv2.FONT_HERSHEY_SIMPLEX, font_size, annot_color, font_scale)
    cv2.putText(image, 'right', (img_width//2 + img_width//4 - cv2.getTextSize('right', cv2.FONT_HERSHEY_SIMPLEX, font_size, font_scale)[0][0], img_height-10), cv2.FONT_HERSHEY_SIMPLEX, font_size, annot_color, font_scale)
    # cv2.imwrite('combined.png', image)
    return image

def pano_centered_viewing(cur_ang, view_num=36):
    """
    Roll the panoramic image to center the view on cur_ang.
    
    Args:
        cur_ang (float): Target angle.
        view_num (int): Number of views in the panoramic image.
    Returns:
        rgb_list (list): List of RGB views after rolling.
        depth_list (list): List of depth views after rolling.
    """
    orientations = list(get_camera_orientations(view_num).keys())
    orientations_36 = np.array([float(value) for value in orientations])
    heading_idx = np.argmin(np.abs(orientations_36 - cur_ang))
    # Calculate offset to center the front view
    n = len(orientations_36)
    center_idx = n // 2
    shift = center_idx - heading_idx
    # Use np.roll to shift the array
    fixed_orientations = np.roll(orientations_36, shift)
    rgb_list = [f"rgb_{ang}" for ang in fixed_orientations]
    depth_list = [v.replace("rgb", "depth") for v in rgb_list]
    return shift,rgb_list,depth_list

def all_step_observation(dir, instruction,file_name = "all_steps.png"):
    """
    Combine all step_*.png files under the path env-img/{traj}/step_view/,
    such as step_1.png, step_2.png into one image.
    Concatenate according to the number order after underscore, and annotate the corresponding filename
    as subtitle in the upper left corner of each sub-image.
    Sub-images are arranged from left to right and top to bottom, with at most 5 files per row,
    and equal spacing around each file.
    
    Args:
        dir (str): Root directory.
        traj (str): Trajectory name.
        file_name (str): Filename to save the final concatenated image.
    """
    # Define path
    step_dir = os.path.join(dir,"step_view")

    # Get all filenames that meet the conditions
    files = [f for f in os.listdir(step_dir) if f.startswith("step") and f.endswith(".png")]
    # Sort by number after underscore
    # Use regex to extract number after step
    files = sorted(files, key=lambda x: int(re.search(r"step(\d+)", x).group(1)))
    # Read all images
    images = [cv2.imread(os.path.join(step_dir, f)) for f in files]

    # Check image reading
    if not images:
        logger.info("No images found that meet the conditions.")
        return None
    # Maximum number of images per row
    max_per_row = 5

    # Sub-image spacing
    margin = 10

    # Get width and height of single sub-image (assuming all sub-images are the same size)
    img_h, img_w = images[0].shape[:2]

    # Calculate grid layout
    num_images = len(images)
    rows = ceil(num_images / max_per_row)
    cols = min(max_per_row, num_images)

    # Create canvas
    canvas_h = rows * img_h + (rows + 1) * margin
    canvas_w = cols * img_w + (cols + 1) * margin
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Draw images on canvas
    for idx, (image, file) in enumerate(zip(images, files)):
        row = idx // max_per_row
        col = idx % max_per_row

        x_start = margin + col * (img_w + margin)
        y_start = margin + row * (img_h + margin)

        # Place image on canvas
        canvas[y_start:y_start + img_h, x_start:x_start + img_w] = image
        # Add title text
        text = file.replace(".png",'').replace("_"," ")
        cv2.putText(canvas, text, (x_start + 5, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Save concatenated image
    output_path = os.path.join(step_dir, file_name)
    canvas = draw_instruction_to_image(canvas,instruction,output_path,save_pics_local = False)
    cv2.imwrite(output_path, canvas)
    # logger.info(f"Concatenated image saved to: {output_path}")

def check_output_path(dir,sub_dir,file_name):
    new_dir = os.path.join(dir,sub_dir)
    os.makedirs(new_dir,exist_ok=True)
    file_path = os.path.join(new_dir,file_name)
    return file_path

def tensor_to_serializable(obj):
    """
    Recursively convert Tensor or ndarray in objects to JSON serializable format.
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        # Convert ndarray to list
        return obj.tolist()
    elif isinstance(obj, np.int64) or isinstance(obj, np.float64):
        # Convert numpy int64 or float64 to Python native types
        return obj.item()
    elif isinstance(obj, dict):
        # Recursively process each key-value pair in the dictionary
        return {key: tensor_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Recursively process each element in the list
        return [tensor_to_serializable(item) for item in obj]
    else:
        # For other types, return directly
        return obj

def determine_position(heading, ang):
    # Counterclockwise, calculate angle difference
    angle_diff = ang - heading
    # Normalize angle difference to [-π, π] range
    angle_diff = (angle_diff + 180) % 360 - 180
    # Determine target position
    if -25 < angle_diff < 25:
        return "forward"
    elif 25 <= angle_diff < 135:
        return "left"
    elif -135 < angle_diff <= -25:
        return "right"
    else:
        return "back"
# Define coordinate conversion function
def polar_to_pixel(theta, hfov, distance, img_width, img_height):
    """Convert polar coordinates (angle and distance) to pixel coordinates of panoramic image"""
    x = int((theta / hfov) * img_width)  # Convert radians to horizontal coordinates
    y = int((1 - distance / 10) * img_height)  # Convert distance to vertical coordinates
    return x % img_width, y

def polar_to_pixel_realDeploy(theta, hfov, distance, img_width, img_height, camera_height=1.5):
    """Convert polar coordinates (angle and distance) to pixel coordinates of panoramic image"""
    x = int((theta / hfov) * img_width)  # Convert radians to horizontal coordinates
    y_theta = math.atan(distance / camera_height)
    if y_theta < 0:
        y_theta = y_theta + math.pi
    y = int((1 - y_theta / math.pi) * img_height)  # Convert distance to vertical coordinates
    return x % img_width, y

def get_pano_slice_indices(cur_ang, orientations, num_slices=36):
    """
    Calculate all slice indices in the left, right, back, and front ranges of the panoramic image
    based on the current angle `cur_ang`, used for panoramic image stitching. When located on the same side
    but split by 0 and 360 degrees, sort according to the method of 0, 360 degrees in the middle.
    """
    heading = math.degrees(cur_ang)
    # Calculate angle interval for each slice
    angle_interval = math.degrees(2 * np.pi / num_slices)  # Angle interval for each slice
    side_ids =  {'left':[],'forward':[],'right':[],'back':[]}
    # Calculate slice index where current angle is located
    # logger.info(f"heading:{heading}")
    for orientation in orientations:
        ang = float(orientation)  # Center angle of the slice
        side = determine_position(heading=heading, ang=ang+ angle_interval / 2)
        side_ids[side].append(ang)  # Store angle instead of index
    # Sort slices for each direction
    # Sort slices for each direction
    for side in side_ids:
        if side_ids[side]:
            # Handle splitting by 0 and 360 degrees
            max_ang = max(side_ids[side])
            min_ang = min(side_ids[side])
            # logger.info(f"{min},{max}")
            if (max_ang-min_ang)>180:
                # Put angles greater than 360 first, then 360, finally 0 and angles greater than 0
                greater_than_180 = [x for x in side_ids[side] if x > 180]
                less_equal_180 = [x for x in side_ids[side] if x <= 180]
                greater_than_180.sort(reverse=True)
                less_equal_180.sort()
                # Concatenate sorted results
                side_ids[side] = greater_than_180 + less_equal_180
            else:
                side_ids[side] = sorted(side_ids[side])  # Normal sorting for other directions
        side_ids[side] = [f"rgb_{ang}" for ang in side_ids[side]]  # Add rgb prefix to elements
    # Return results
    return side_ids

def draw_heading_on_image(image, cur_ang=0, hfov=360, save_pics_local=False, output_path="output.png"):
    """
    Draw a green arrow indicating current heading on the panoramic image, and display "heading" text
    centered above the arrow.
    
    Args:
        image (ndarray): Input image (panoramic image or other).
        cur_ang (float): Current heading angle in radians (0 to 2*pi).
        hfov (int): Horizontal field of view angle (usually 360 degrees for panoramic).
        save_pics_local (bool): Whether to save the result image to local file.
        output_path (str): Output file path if saving.
    
    Returns:
        ndarray: Output image containing arrow and label.
    """
    img_height, img_width = image.shape[:2]
    rect_width = img_width // 8
    output_image = image.copy()
    heading = math.degrees(cur_ang)
    # Calculate current heading position in the image
    # x_labels = ['left','ahead','right','behind']
    # angs = [heading-90,heading,heading+90,heading+180]
    x_labels = ['heading']
    angs = [heading]
    for ang, label in zip(angs, x_labels):
        # Arrow parameters
        x = int((ang%360 / hfov) * img_width)
        arrow_length = 40  # Arrow length
        arrow_thickness = 6  # Arrow thickness
        arrow_head_length = 10  # Arrow head length

        # Arrow start point (x coordinate, y coordinate at bottom axis position)
        if label == 'back':
            # For 'behind' direction, draw semi-transparent rectangle
            # Calculate rectangle area width, rectangle covers both sides of image
            
            left_rect = (0, 0, rect_width, img_height)
            right_rect = (img_width - rect_width, 0, rect_width, img_height)

            # Semi-transparent rectangle overlay
            overlay = output_image.copy()
            cv2.rectangle(overlay, (left_rect[0], left_rect[1]), (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), (255, 255, 255), -1)
            cv2.rectangle(overlay, (right_rect[0], right_rect[1]), (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]), (255, 255, 255), -1)

            # Create semi-transparent effect
            cv2.addWeighted(overlay, 0.6, output_image, 1 - 0.6, 0, output_image)

            # Draw "behind" text above rectangle area
            # text = label
            # text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            # text_x = (img_width - text_size[0]) // 2  # Center text
            # text_y = 30  # Text position

            # # Draw text
            # cv2.putText(output_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            start_point = (x, img_height - 90)
            # Arrow end point
            end_point = (x, img_height - 90 - arrow_length)

            # Draw arrow
            cv2.arrowedLine(output_image, start_point, end_point, (0, 255, 0), arrow_thickness, tipLength=0.1)

            # Draw "heading" text above arrow head
            text = label
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = max(0, min(x - text_size[0] // 2, img_width - text_size[0]))
            text_y = end_point[1] - arrow_head_length - 10  # Text position above arrow head
            
            # Draw text
            cv2.putText(output_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    # Save result image
    if save_pics_local:
        cv2.imwrite(output_path, output_image)
        # logger.info(f"Image saved to: {output_path}")
    return output_image

def get_pano(observations,VIEW_NUM,RGB_LIST,DEPTH_LIST, need_depth=False):
    IMAGE_SIZE = 256
    SPLIT_NUM = 90 / (360/VIEW_NUM)
    OFFSET = IMAGE_SIZE*200/256/VIEW_NUM+2 # 6
    ROTATE_CENTER = False
    pano_rgb = []
    pano_depth = []
    
    rgb_frame = []
    depth_frame = []
    for i in range(VIEW_NUM-1,-1,-1):
        slice = center_crop(observations[RGB_LIST[i]][:,:,:3],  (int(IMAGE_SIZE/SPLIT_NUM - OFFSET), IMAGE_SIZE))
        if isinstance(slice,torch.Tensor):
            slice= slice.cpu().numpy()
        rgb_frame.append(slice)
        if need_depth:
            depth = (observations[DEPTH_LIST[i]].squeeze() * 255)
            if isinstance(depth,torch.Tensor):
                depth = depth.cpu().numpy().astype(np.uint8)
            depth = np.stack([depth for _ in range(3)], axis=2)
            depth = center_crop(depth, (int(IMAGE_SIZE/SPLIT_NUM - OFFSET), IMAGE_SIZE))
            depth_frame.append(depth)
    pano_rgb = np.concatenate(rgb_frame, axis=1)
    if ROTATE_CENTER:
        pano_rgb = np.roll(pano_rgb, pano_rgb.shape[1]//2, axis=1)
    if need_depth:
        pano_depth = np.concatenate(depth_frame, axis=1)
        if ROTATE_CENTER:
            pano_depth = np.roll(pano_depth, pano_depth.shape[1]//2, axis=1)
    else:
        pano_depth = None
    
    return pano_rgb,pano_depth

def draw_text_on_image(image, x_pos, text, arrow_length=100, font_scale=0.5, font_thickness=1, color=(0, 255, 0), line_spacing=2):
    start_y = int(image.shape[0])  # Image height
    start_x = x_pos
    end_x = int(start_x)
    end_y = int(start_y - arrow_length)
    # Draw arrow
    image = cv2.arrowedLine(image, (start_x, start_y), (end_x, end_y), color, thickness=font_thickness, tipLength=0.1)
    # Calculate text position
    text_x = int(end_x + 5)
    text_y = int(end_y - 30)
    # Draw text
    for i, line in enumerate(text.split('\n')):
        y_offset = int(i * line_spacing * font_scale * 20)  # Calculate y offset for each line of text
        cv2.putText(image, line, (text_x, text_y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
    return image

def draw_waypoint_on_pano(pano_image, cand_vp, shift, current_step_id, cand_vp_step_id, hfov=360, center_as_zero=False):
    img_height, img_width = pano_image.shape[:2]
    output_image = pano_image.copy()
    def draw_marker(output_image,x,y,vp):
        # Draw red point
        cv2.circle(output_image, (x, y), 8, (255,0,0), -1)
        # Add text information
        text = f"{vp}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_bg_width = text_size[0]
        text_bg_height = text_size[1]
        
        # Calculate text and background box position, ensure center alignment
        text_x = max(0, min(x - text_size[0] // 2, img_width - text_size[0]))
        text_y = y - text_bg_height  # Text directly above the red point
        
        # Draw semi-transparent background, background box centered with text
        overlay = output_image.copy()
        cv2.rectangle(overlay, (text_x, text_y - text_bg_height), 
                    (text_x + text_bg_width, text_y + 5), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.6, output_image, 0.4, 0, output_image)
        cv2.putText(output_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return output_image
    for vp,vp_info in cand_vp.items():
        if cand_vp_step_id[vp] != current_step_id:
            # only draw the candidate point of the current step
            continue
        point_angle = vp_info['polar'][0]
        point_angle_deg = math.degrees(point_angle)
        point_distance = vp_info['polar'][1]
        if center_as_zero:
            # Use center as 0 degrees, counterclockwise as positive
            if point_angle_deg < 180 and point_angle_deg > 0:
                offset = 180 - point_angle_deg
            else:
                offset = 180 - point_angle_deg
            x, y = polar_to_pixel_realDeploy(offset, distance=point_distance, hfov=hfov, img_width=img_width, img_height=img_height)
        else:
            # Original calculation method
            offset = hfov-(point_angle_deg +shift*10)%hfov
            x, y = polar_to_pixel(offset, distance=point_distance, hfov=hfov, img_width=img_width, img_height=img_height)
        output_image = draw_marker(output_image,x,y,vp)
        # if x < img_width // 8 or x > img_width * 7 // 8:
        #     output_image = draw_marker(output_image,img_width - x, y, vp)
    return output_image

def pad_and_stack_images_vertically(image1, image2):
    # Get width of two images
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape

    # Determine target width as the larger width
    target_width = max(width1, width2)
    # Calculate left and right padding needed
    pad_left1 = (target_width - width1) // 2
    pad_right1 = target_width - width1 - pad_left1

    pad_left2 = (target_width - width2) // 2
    pad_right2 = target_width - width2 - pad_left2

    # Add left and right white padding to two images
    padded_image1 = cv2.copyMakeBorder(image1, 0, 0, pad_left1, pad_right1, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    padded_image2 = cv2.copyMakeBorder(image2, 0, 0, pad_left2, pad_right2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # Vertically stack two images
    stacked_image = np.vstack([padded_image1, padded_image2])
    return stacked_image

def concatenate_images_with_labels(images, labels, font_scale=1, thickness=2):
    # Get height and width of single image
    height, width, _ = images[0].shape #224，224，3
    spacing = 4
    # Calculate uniform spacing between images
    num_images = len(images)
    # Set target width and total height (including image and label areas)
    total_width = width*num_images+spacing*(num_images+1)
    total_height = height + 50
    # Create a white background image for concatenation
    concatenated_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    # Concatenate images and add labels
    x_offset = spacing  # Initial offset
    for img, label in zip(images, labels):
        # Place image
        if isinstance(img,torch.Tensor):
            img = img.cpu().numpy()
        concatenated_image[:height, x_offset:x_offset + width] = img

        # Add label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x_offset + (width - text_size[0]) // 2
        text_y = height + text_size[1] + 10
        cv2.putText(concatenated_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Update offset
        x_offset += width + spacing

    return concatenated_image

def draw_instruction_to_image(pano, instruction, output_path,save_pics_local = True, gt_instr=None):
    """
    Add instruction text to the bottom and save as a new image.
    
    Args:
        pano (ndarray): Original image.
        instruction (str): Instruction text to add to the image.
        output_dir (str): Output image folder path.
        max_step_id (int): Current step ID, used for naming when saving image.
    """
    # Get image width and height
    h, w, c = pano.shape
    
    # Set font and calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (0, 0, 0)
    bg_color = (255, 255, 255)
    
    # Split instruction text into multiple lines
    if gt_instr is not None:
        concat_instruction = instruction + gt_instr
    else:
        concat_instruction = instruction

    words = concat_instruction.split(' ')
    lines = []
    current_line = words[0]
    for word in words[1:]:
        text_size = cv2.getTextSize(current_line + ' ' + word, font, font_scale, thickness)[0]
        if text_size[0] > w - 20:  # 20 pixels padding
            lines.append(current_line)
            current_line = word
        else:
            current_line += ' ' + word
    lines.append(current_line)

    # Create a white background image
    line_height = cv2.getTextSize('Tg', font, font_scale, thickness)[0][1] + 10
    bg_height = line_height * len(lines) + 10  # Background height slightly larger than text height
    background = np.full((bg_height, w, 3), bg_color, dtype=np.uint8)

    # Draw text on white background
    for idx, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = 10  # Text x coordinate
        text_y = (idx + 1) * line_height  # Text y coordinate
        cv2.putText(background, line, (text_x, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    # Concatenate white background with original image
    pano_with_instruction = np.vstack((background,pano))
    if save_pics_local:
        # Generate output image path
        cv2.imwrite(output_path, pano_with_instruction)
        # logger.info(f"Image saved to: {output_path}")
    return pano_with_instruction

def draw_gpt_to_image(pano, instruction, output_path,save_pics_local = True):
    """
    Add instruction text to the bottom and save as a new image.
    
    Args:
        pano (ndarray): Original image.
        instruction (str): Instruction text to add to the image.
        output_dir (str): Output image folder path.
        max_step_id (int): Current step ID, used for naming when saving image.
    """
    # Get image width and height
    h, w, c = pano.shape
    
    # Set font and calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (0, 0, 0)
    bg_color = (255, 255, 255)
    
    # Split instruction text into multiple lines
    words = instruction.replace('\n',' #').split(' ')
    lines = []
    current_line = words[0]
    for word in words[1:]:
        text_size = cv2.getTextSize(current_line + ' ' + word, font, font_scale, thickness)[0]
        if text_size[0] > w - 20:  # 20 pixels padding
            lines.append(current_line)
            current_line = word
        else:
            current_line += ' ' + word
    lines.append(current_line)

    # Create a white background image
    line_height = cv2.getTextSize('Tg', font, font_scale, thickness)[0][1] + 10
    bg_height = line_height * len(lines) + 10  # Background height slightly larger than text height
    background = np.full((bg_height, w, 3), bg_color, dtype=np.uint8)

    # Draw text on white background
    for idx, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = 10  # Text x coordinate
        text_y = (idx + 1) * line_height  # Text y coordinate
        cv2.putText(background, line, (text_x, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    # Concatenate white background with original image
    pano_with_instruction = np.vstack((pano,background))
    if save_pics_local:
        # Save the composite image
        cv2.imwrite(output_path, pano_with_instruction)
        # logger.info(f"Image saved to: {output_path}")
    return pano_with_instruction
# Usage example

if __name__ == "__main__":
    pano_path = "pano.png"  # Input panoramic image path
    points_info = [
        {"name": "g0", "theta": 0, "distance": 1.0},
        {"name": "g1", "theta": np.pi / 2, "distance": 1.5},
        {"name": "g2", "theta": np.pi, "distance": 0.5},
        {"name": "g3", "theta": 3 * np.pi / 2, "distance": 1.2}
    ]
    output_dir = "./"  # Output directory
    angle_interval = 30  # Angle interval
