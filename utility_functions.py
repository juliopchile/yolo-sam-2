import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def draw_boxes(image, boxes):
    """
    Draws one or more boxes on the image.

    Parameters:
        image (numpy.ndarray): The original image in RGB format. Shape: (H, W, 3).
        boxes (list or tuple): A list of boxes or a single box. 
                               Each box is a tuple of (x0, y0, x1, y1).

    Returns:
        numpy.ndarray: The image with the boxes drawn.
    """
    output_image = image.copy()

    # If boxes is None or empty, return the original image
    if boxes is []:
        return output_image

    # If a single box is provided, convert it to a list of one box
    if isinstance(boxes, tuple):
        boxes = [boxes]

    # Draw each box on the image
    for box in boxes:
        x0, y0 = int(box[0]), int(box[1])
        x1, y1 = int(box[2]), int(box[3])
        width, height = x1 - x0, y1 - y0
        
        # Draw the rectangle with the given coordinates and size
        output_image = cv2.rectangle(output_image, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), thickness=2)

    return output_image
