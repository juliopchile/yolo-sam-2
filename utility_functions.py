import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ? Utility functions to show results of SAM2 segmentations, using Matplotlib

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


# ? Utility functions to show results of SAM2 segmentations, using OpenCV

def draw_masks_on_image(image, masks, random_color=False, borders=True):
    """
    Draws masks onto the image.

    Parameters:
        image (numpy.ndarray): The original image in RGB format. Shape: (H, W, 3).
        masks (numpy.ndarray): The masks to draw. Shape: (num_masks, batch=1, H, W).
        random_color (bool): If True, use random colors for each mask. Otherwise, use a fixed color.
        borders (bool): If True, draw borders around the masks.

    Returns:
        numpy.ndarray: The image with masks drawn, in RGB format.
    """
    # Make a copy of the image to avoid modifying the original
    output_image = image.copy()
    height, width, _ = output_image.shape

    # Ensure the image is in float format for blending
    output_image = output_image.astype(np.float32)

    for mask in masks:
        # Remove the batch dimension
        mask = mask.squeeze(0)  # Now shape: (H, W)

        # Define the color for the mask
        if random_color:
            # Generate a random color
            color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        else:
            # Use a fixed color (e.g., [30, 144, 255] in RGB)
            color = np.array([30, 144, 255], dtype=np.uint8)

        # Define the transparency factor
        alpha = 0.6

        # Create a colored version of the mask
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        colored_mask[mask == 1] = color

        # Blend the colored mask with the original image
        # Identify the regions where the mask is applied
        mask_indices = mask == 1
        mask_indices_3d = np.repeat(mask_indices[:, :, np.newaxis], 3, axis=2)

        # Apply blending
        output_image[mask_indices_3d] = (alpha * colored_mask[mask_indices_3d] + (1 - alpha) * output_image[mask_indices_3d])

        if borders:
            # Find contours in the mask
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Smooth the contours
            smoothed_contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]

            # Draw the contours on the image
            cv2.drawContours(output_image, smoothed_contours, -1, (255, 255, 255), thickness=2)

    # Convert the image back to uint8 format
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    return output_image

def draw_points(image, coords, labels, marker_size=15):
    output_image = image.copy()
    for coord, label in zip(coords, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.drawMarker(output_image, tuple(coord), color, markerType=cv2.MARKER_STAR, markerSize=marker_size, thickness=2, line_type=cv2.LINE_AA)
    return output_image

def draw_boxes(image, boxes, line_color=(0, 255, 0)):
    """
    Draws one or more boxes on the image.

    Parameters:
        image (numpy.ndarray): The original image in RGB format. Shape: (H, W, 3).
        boxes (list or tuple): A list of boxes or a single box. 
                               Each box is a tuple or array of (x0, y0, x1, y1).

    Returns:
        numpy.ndarray: The image with the boxes drawn.
    """
    output_image = image.copy()

    # If boxes is None or empty, return the original image
    if not boxes:
        return output_image

    # Handle single box case
    if isinstance(boxes, (tuple, list)) and len(np.shape(boxes)) == 2:
        boxes = [boxes]

    # Flatten the list if the boxes were provided as an array within a list
    if isinstance(boxes, list) and len(boxes) == 1 and isinstance(boxes[0], np.ndarray):
        boxes = boxes[0]

    # Draw each box on the image
    for box in boxes:
        # Extract coordinates
        if isinstance(box, np.ndarray):
            x0, y0, x1, y1 = box
        else:
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        
        # Draw the rectangle with the given coordinates
        output_image = cv2.rectangle(output_image, (x0, y0), (x1, y1), line_color, thickness=2)

    return output_image


# ? Utility functions to show results of YOLO segmentations, using OpenCV

def resize_yolo_masks_tensor(segmentation_result):
    """
    Resizes YOLO segmentation masks to their original image shape.

    This function takes a segmentation result from a YOLO model that contains masks,
    resizes each mask to the original shape of the image from which it was predicted,
    and returns the resized masks as a NumPy array of type uint8.

    Parameters:
    segmentation_result (list): A list of segmentation result objects. Each object should
                                have an attribute `masks` with two properties:
                                - `data`: A torch tensor of shape (N, H, W) where N is the 
                                          number of masks, H is the height, and W is the width.
                                - `orig_shape`: A tuple (original_height, original_width) which
                                                represents the original shape of the image.

    Returns:
    np.ndarray: A NumPy array of resized masks of shape (N, original_height, original_width) and type uint8.
                Each mask is resized to the original dimensions of the image.
                
    Example:
    segmentation_result = model(input)
    resized_masks = resize_yolo_masks_tensor(segmentation_result)
    """

    original_shape = segmentation_result[0].masks.orig_shape
    masks = segmentation_result[0].masks.data  # masks.shape = torch.Size([N, H, W]) = (mask_number, height, width)
    
    # Reshape the masks tensor to add a channel dimension which is needed for F.interpolate
    masks = masks.unsqueeze(1)  # masks.shape becomes torch.Size([N, 1, H, W])
    
    # Interpolate the masks to the original_shape
    resized_masks = F.interpolate(masks, size=original_shape, mode='nearest')
    
    # Remove the channel dimension
    resized_masks = resized_masks.squeeze(1)  # resized_masks.shape becomes torch.Size([N, original_shape[0], original_shape[1]])
    
    # Convert the resized masks to a NumPy array and ensure type uint8
    resized_masks = resized_masks.cpu().numpy().astype(np.uint8)
    
    return resized_masks

def draw_yolo_mask(canvas, masks, random_color=False, borders=True):
    """
    Draws the segmentation masks onto the canvas image, resizing them if necessary.

    Args:
        canvas (numpy.ndarray): The image on which to draw the masks.
        masks (torch.Tensor): A tensor of shape [number of masks, height, width] containing the mask data.
        random_color (bool): If True, each mask will be drawn with a random color. Default is False.
        borders (bool): If True, borders will be drawn around the masks. Default is True.
    
    Returns:
        numpy.ndarray: The canvas image with the masks drawn.
    """
    # Get the dimensions of the canvas
    canvas_height, canvas_width = canvas.shape[:2]

    # Loop over each mask
    for mask in masks:
        
        # Resize the mask to fit the canvas size
        resized_mask = cv2.resize(mask, (canvas_width, canvas_height), interpolation=cv2.INTER_NEAREST)
        
        if random_color:
            color = np.random.randint(0, 256, size=3)  # Generate a random color
        else:
            color = (0, 255, 0)  # Default green color
        
        # Create a color image for the mask
        colored_mask = np.zeros_like(canvas, dtype=np.uint8)
        colored_mask[resized_mask > 0] = color
        
        # Overlay the mask onto the canvas
        canvas = cv2.addWeighted(canvas, 1.0, colored_mask, 0.5, 0)
        
        if borders:
            # Find contours around the mask
            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(canvas, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    
    return mask_image

def draw_sam2_mask(canvas, masks, random_color=False, borders=True):
    """
    Draws the segmentation masks onto the canvas image.
    Args:
        canvas (numpy.ndarray): The image on which to draw the masks.
        masks (numpy.ndarray): A numpy array of shape [num_predicted_masks_per_input, height, width]
                               or [num_predicted_masks_per_input, batch_size, height, width] containing the mask data.
        random_color (bool): If True, each mask will be drawn with a random color. Default is False.
        borders (bool): If True, borders will be drawn around the masks. Default is True.
    
    Returns:
        numpy.ndarray: The canvas image with the masks drawn.
    """
    
    def process_mask(mask):
        nonlocal result_canvas
        if random_color:
            color = np.random.randint(0, 256, size=3)
        else:
            color = [30, 144, 255]  # SAM2 default color
        
        alpha = 0.6
        colored_mask = np.zeros_like(result_canvas, dtype=np.uint8)
        colored_mask[mask > 0] = color
        
        result_canvas = cv2.addWeighted(result_canvas, 1.0, colored_mask, alpha, 0)

        if borders:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(result_canvas, contours, -1, (255, 255, 255), thickness=2)
    
    result_canvas = canvas.copy()
    
    if masks.ndim == 3:
        # Single bbox case: [num_predicted_masks_per_input, height, width]
        for mask in masks:
            process_mask(mask)
    elif masks.ndim == 4:
        # Multiple bboxes case: [num_predicted_masks_per_input, batch_size, height, width]
        # Process each mask in batch
        for batch_idx in range(masks.shape[1]):
            for mask in masks[:, batch_idx]:
                process_mask(mask)
    else:
        raise ValueError("Unexpected masks dimensions. Expected 3D or 4D input.")
    
    return result_canvas


# ? Utility functions to get values from YOLO restuls

def get_bboxes(results):
    """ Returns a list of boxes from a YOLO result frame """
    list_of_bounding_boxes = []
    for result in results:
        bounding_boxes_in_frame = np.array([bb.cpu().numpy() for bb in result.boxes.xyxy], dtype=np.int16)
        list_of_bounding_boxes.append(bounding_boxes_in_frame)
    return list_of_bounding_boxes

def get_contours(results):
    """ Returns a list of contours from a YOLO result frame """
    list_of_contours = []
    for result in results:
        contours_in_frame = result.masks.xy
        list_of_contours.append(contours_in_frame)
    return list_of_contours

def find_centroid(contour):
    """ Return the centroid of a given contour as a list of two values [x, y] """
    centroid = np.mean(contour, axis=0)
    return centroid

def get_centroids(list_of_contours):
    """ Return a list of centroids tuples, given a list of contours """
    list_of_points = []
    for frame_contours in list_of_contours:
        centroids_in_frame = []
        for contour in frame_contours:
            centroids_in_frame.append(find_centroid(contour))
        list_of_points.append(centroids_in_frame)

    return list_of_points

def find_point_inside_contour(contour, number_of_points):
    # TODO make the implementation
    # Find a number of points as a pairs of (x,y) values.
    # These points need to be inside the mask that is formed by the contour (inside the area).
    # Points need to be in the next format np.array([[x1, y1], [x2, y2],...], dtype=np.float32)
    # for example for a single point: points = np.array([[820, 500]], dtype=np.float32)
    pass

def get_points_inside_contours(list_of_contours, number_of_points):
    list_of_points = []
    for frame_contours in list_of_contours:
        points_in_frame = []
        for contour in frame_contours:
            points_in_contour = find_point_inside_contour(contour, number_of_points)
            points_in_frame.append(points_in_contour)
        list_of_points.append(points_in_contour)

    return list_of_points

def generate_random_points_outside_boxes(num_points, image_size, bounding_boxes):
    height, width = image_size
    points = []

    def is_point_outside_boxes(point, boxes):
        x, y = point

        # Ensure boxes is in the correct format (array of shape (n, 4))
        if isinstance(boxes, list):
            boxes = np.vstack(boxes)

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return False
        return True

    while len(points) < num_points:
        #random_point = (random.randint(0, width - 1), random.randint(0, height - 1))
        random_point = (np.random.randint(0, width), np.random.randint(0, height))
        if is_point_outside_boxes(random_point, bounding_boxes):
            points.append(random_point)

    return points
