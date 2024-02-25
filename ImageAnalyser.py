import numpy as np
from ultralytics import YOLO
import cv2
import uuid

# Initialize the YOLO model outside the function
model = YOLO("models/yolov8m-seg.pt")



def get_segmentation_data(model, img_path):
    """
    Get segmentation data from the given model and image path. 

    Parameters:
    model (object): The model used for prediction.
    img_path (str): The path to the image.

    Returns:
    bboxes (np.array): The bounding boxes.
    class_ids (np.array): The class ids.
    segmentation_contours_idx (list): The segmentation contours.
    scores (np.array): The scores.
    labels (list): The labels.
    confs (list): The confidences.
    """
    # Read and resize the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.7, fy=0.7)

    # Predict the image using the model
    results = model.predict(source=img.copy(), save=False, save_txt=False)
    result = results[0]

    # Get the shape of the image
    height, width, _ = img.shape

    # Extract the labels from the detections
    names = model.names
    labels = [names[int(i)] for i in result.boxes.cls]
    confs = [float(i) for i in result.boxes.conf]

    segmentation_contours_idx = []
    for seg in result.masks.segments:
        # Adjust the contours to the actual image size
        seg[:, 0] *= width
        seg[:, 1] *= height
        segment = np.array(seg, dtype=np.int32)
        segmentation_contours_idx.append(segment)

    # Get the bounding boxes, class ids and scores
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
    scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

    return bboxes, class_ids, segmentation_contours_idx, scores, labels, confs



def generate_segmented_image(image_path, mode='polyline', filter_duplicates=True):
    """
    Generate a segmented image from the given image path. 

    Parameters:
    image_path (str): The path to the image to be segmented.
    mode (str): The mode for drawing. It could be 'bbox', 'polyline', or 'both'. Default is 'both'.
    filter_duplicates (bool): If True, return only unique labels. If False, return all labels. Default is True.

    Returns:
    labels (list): List of labels for the segmented areas in the image.
    output_image_path (str): The path to the output image with bounding boxes.
    """
    # Get segmentation data from the model
    bboxes, classes, segmentations, scores, labels, confs = get_segmentation_data(model, image_path)

    # Read and resize the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=0.7, fy=0.7)

    # Iterate over each bbox, class_id, seg, score, label, conf
    for bbox, class_id, seg, score, label, conf in zip(bboxes, classes, segmentations, scores, labels, confs):
        (x, y, x2, y2) = bbox
        # Draw bounding box if mode is 'bbox' or 'both'
        if mode in ['bbox', 'both']:
            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 1)
        # Draw polyline if mode is 'polyline' or 'both'
        if mode in ['polyline', 'both']:
            cv2.polylines(img, [seg], True, (0, 0, 255), 2)
        # Put label and confidence text on the image
        cv2.putText(img,
                    label + ' ' + str(round(conf, 2)),
                    (x, y - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 0, 0),
                    thickness=1
                    )

    # Generate a unique filename for the output image
    unique_filename = str(uuid.uuid4()) + '-2.jpeg'
    # Set the output image path
    output_image_path = 'static/' + unique_filename
    # Save the image
    cv2.imwrite(output_image_path, img)

    # Filter duplicate labels if filter_duplicates is True
    if filter_duplicates:
        labels = list(set(labels))

    # Return labels and output image path
    return labels, output_image_path
