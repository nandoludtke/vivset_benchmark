import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

# model_path = 'best.pt'
# model = YOLO(model_path)

class LabelPredictor:
    def __init__(self, model_path: str, relevant_items: list=[], skip=False):
        abs_model_path = os.path.abspath(model_path)
        if not os.path.exists(abs_model_path):
            print(f"Model path {abs_model_path} is not valid.")
            return
        self.model = YOLO(abs_model_path)
        self.relevant_items = relevant_items
        self.skip = skip

    def label_images(self, images_list: list, skip=False):
        # input handling
        if not images_list:
            print("Image list is empty.")
            return
        # create temp folder for labeled images
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(images_list[0])), 'temp')
        if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
        if not os.path.exists(temp_dir) or not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
            print(f"Temp dir created at {temp_dir}")
        else:
            print(f"Using temp dir at {temp_dir}")
        # iterate through images paths
        image_item_list = []
        for image_path in images_list:
            print(f"Getting labels for {image_path}")
            abs_image_path = os.path.abspath(image_path)
            # check if file is available
            if not os.path.exists(abs_image_path) or not os.path.splitext(abs_image_path)[1].lower() in ['.jpeg', '.jpg', '.png']:
                print(f"Image path {image_path} does not exist. Deleting temp folder.")
                shutil.rmtree(temp_dir)
                return
            # start labeling
            # open image
            opened_img = cv2.imread(abs_image_path)
            # resize image to width of 640px
            img_height, img_width, _ = opened_img.shape
            factor = 640 / img_width
            factor = 512 / img_width
            dimension = (int(img_width * factor), int(img_height * factor))
            img = cv2.resize(opened_img, dimension)
            print(f"Image was resized to x: {img.shape[1]} and y: {img.shape[0]}")
            if not self.skip:
                # predict bounding boxs
                bounding_boxes = self.get_boxes(
                    image=img,
                    relevant_items=self.relevant_items
                )
                # add bounding boxes and labels to image
                labeled_img = self.draw_box_label(
                    image=img,
                    bounding_boxes=bounding_boxes
                )
                # add found labels to image item list
                image_item_list.append(
                    {
                        "image": image_path,
                        "found_items": [box['item'] for box in bounding_boxes]
                    }
                )
            else:
                labeled_img = img
            if not cv2.imwrite(os.path.join(temp_dir, os.path.basename(image_path)), labeled_img):
                print(f"Could not write image to {os.path.join(temp_dir, os.path.basename(image_path))}")
            print(f"Labeled image saved to {os.path.join(temp_dir, os.path.basename(image_path))}")
        return image_item_list

    def predict_position(self, img):
        H, W, _ = img.shape
        results = self.model(img)
        predicted_classes = results[0].boxes.cls
        confidance_values = results[0].boxes.conf
        coordinates = results[0].boxes.xyxy
        index = 0
        dictObjectPosition = {0: ['disinfectant_wipe'], 1: ['face_mask_box'], 2: ['face_shield'], 3: ['gloves'], 4: ['gown'],
                              5: ['hand_sanitizer'], 6: ['n95_box'], 7: ['n95_mask'], 8: ['paper_towel'],
                              9: ['poster'], 10: ['sink_handle'], 11: ['stethoscope'], 12: ['trash_can'], 13: ['water_stream']}
        for obj in predicted_classes:
            obj_int = int(obj)
            if len(dictObjectPosition[obj_int]) == 1:
                dictObjectPosition[obj_int].append(coordinates[index].tolist())
                dictObjectPosition[obj_int].append(float(confidance_values[index]))
            elif len(dictObjectPosition[obj_int]) > 2:
                if (float(confidance_values[index]) > dictObjectPosition[obj_int][2]) | (coordinates[index].tolist()[2] > dictObjectPosition[obj_int][1][2]) :
                    dictObjectPosition[obj_int][1] = coordinates[index].tolist()
                    dictObjectPosition[obj_int][2] = confidance_values[index]
            index +=1
        return dictObjectPosition
    
    def get_boxes(self, image, relevant_items=[]):
        bounding_boxes = []
        letter_idx = 1
        for _, element in self.predict_position(image).items():
            # get label letter
            letter = chr(ord('@') + letter_idx)
            # check for not allowed letters: J, Q, V
            if letter in ['I', 'Q', 'V']:
                    letter_idx = letter_idx + 1
                    letter = chr(ord('@') + letter_idx)
            if len(element) > 1 and (element[0] in relevant_items or not relevant_items):
                # print(f"Item '{element[0]}' was found with label {letter}")
                # get upper-left and lower-right coordinates
                upper_left = (int(element[1][0]), int(element[1][1]))
                lower_right = (int(element[1][2]), int(element[1][3]))
                # create random color with offset of 100 (prevent dark colors)
                color = list(np.random.uniform(low=100.0, high=256.0, size=3))
                bounding_boxes.append(
                    {
                        'mark': letter,
                        'item': element[0],
                        'upper_left': upper_left,
                        'lower_right': lower_right,
                        'color': color
                    }
                )
            letter_idx = letter_idx + 1
        return bounding_boxes

    def draw_box_label(self, image, bounding_boxes, box_size=(20, 15), text_thickness=1, box_thickness=1):
        for box in bounding_boxes:
            # draw rectangle
            cv2.rectangle(image, box['upper_left'], box['lower_right'], box['color'], thickness=box_thickness)

        # draw labels
        for box in bounding_boxes:
            # create different label positions
            possible_positions = {
                'top_edge_left':        (box['upper_left'][0],                box['upper_left'][1]  - box_size[1]),
                'top_edge_right':       (box['lower_right'][0] - box_size[0], box['upper_left'][1]  - box_size[1]),
                'right_edge_top':       (box['lower_right'][0],               box['upper_left'][1]),
                'right_edge_bottom':    (box['lower_right'][0],               box['lower_right'][1] - box_size[1]),
                'bottom_edge_left':     (box['upper_left'][0],                box['lower_right'][1]),
                'bottom_edge_right':    (box['lower_right'][0] - box_size[0], box['lower_right'][1]),
                'left_edge_top':        (box['upper_left'][0]  - box_size[0], box['upper_left'][1]),
                'left_edge_bottom':     (box['lower_right'][0] - box_size[0], box['upper_left'][1]  - box_size[1])
            }
            for _, position in possible_positions.items():
                top_left = position
                bottom_right = (position[0] + box_size[0], position[1] + box_size[1])

                intersection = False
                for _box in bounding_boxes:
                    # skip same item
                    if _box['item'] == box['item']:
                        continue
                    # check edges for overlap
                    intersection = False
                    if not((_box['lower_right'][0] < top_left[0]) or (_box['upper_left'][0] > bottom_right[0]) or (_box['upper_left'][1] > bottom_right[1]) or (_box['lower_right'][1] < top_left[1])):
                        intersection = True
                        break
                if not intersection:
                    break

            # Draw the filled rectangle on mask with white color and then blend it using the transparency
            cv2.rectangle(image, top_left, bottom_right, box['color'], thickness=-1)
            # Set the font type and scale of text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 0, 0)
            # Calculate the size of the text and the starting position
            label_text = str(box['mark'])
            text_size = cv2.getTextSize(label_text, font, font_scale, text_thickness)[0]
            text_x = top_left[0] + (box_size[0] - text_size[0]) // 2
            text_y = top_left[1] + (box_size[1] + text_size[1]) // 2

            cv2.putText(image, str(label_text), (text_x, text_y), font, font_scale, font_color, text_thickness, cv2.LINE_AA)
        return image