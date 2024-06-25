import cv2
import numpy as np
from rembg import remove
from Person_remover.yolo.detect import prepare_detector
from Person_remover.pix2pix.utils.model import Pix2Pix
from Person_remover.utils import read_image, prepare_image_yolo, cut_result, create_new_image, prepare_frame_p2p, frame_to_int
import matplotlib.pyplot as plt

class Detecion():
    def __init__(self,
                 class_path: str='',
                 config_path: str='',
                 weights_path: str='') -> None:
        """"""
        with open(class_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.COLORS = [[0, 0, 255]] * len(self.classes)
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.scale = 0.00392
        self.conf_threshold = 0.5
        self.nms_threshold = 0.5

        self.yolo = prepare_detector(weights='yolo_remover_weights.h5')
        self.p2p = Pix2Pix(mode='try', checkpoint_dir='Person_remover/pix2pix/checkpoint/')


    def _get_output_layers(self) -> list:
        """"""
        layer_names = self.net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return output_layers


    def _draw_prediction(self,
                         img,
                         class_id: int=0,
                         score: float=0.0,
                         x: int=0,
                         y: int=10,
                         width: int=10,
                         height: int=10,
                         border_width: int=5,
                         show_label=False):
        """"""
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x, y), (width, height), color, border_width)

        if show_label:
            label = str(self.classes[class_id])
            cv2.putText(img, f'{label}: {score}', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 10)


    def detect_objects(self,
                          image=None,
                          image_path=None,
                          person_only=False) -> None:
        """"""
        if image.all() == None:
            try:
                image = cv2.imread(image_path)
            except:
                print('Please provided image data or image path')
                return

        image_width = image.shape[1]
        image_height = image.shape[0]

        blob = cv2.dnn.blobFromImage(image, self.scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self._get_output_layers())

        class_ids = []
        confidence_scores = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    w = int(detection[2] * image_width)
                    h = int(detection[3] * image_height)
                    x = center_x - w/2
                    y = center_y - h/2
                    class_ids.append(class_id)
                    confidence_scores.append(np.round(float(confidence), 3))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, self.conf_threshold, self.nms_threshold)

        if person_only:
            person_boxes = []
            person_indices = []
            person_confidence_scores = []

            for i in indices:
                if class_ids[i] != 0:
                    continue
                else:
                    person_boxes.append(boxes[i])
                    person_indices.append(i)
                    person_confidence_scores.append(confidence_scores[i])

            return [0]*len(person_boxes), person_confidence_scores, person_boxes, person_indices

        return class_ids, confidence_scores, boxes, indices


    def detect_people(self,
                      image=None,
                      image_path=None,
                      path_to_save=None):
        """"""
        if image == None:
            try:
                image = cv2.imread(image_path)
            except:
                print('Please provided image data or image path')
                return

        image_width = image.shape[1]
        border_width = int(image_width/100)

        class_ids, scores, boxes, indicies = self.detect_objects(image=image,
                                                                 person_only=True)
        # print(indicies, boxes)
        for i in range(len(boxes)):
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]

            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            self._draw_prediction(image,
                                  class_ids[i],
                                  scores[i],
                                  round(x),
                                  round(y),
                                  round(x+w),
                                  round(y+h),
                                  border_width,
                                  False)

        cv2.imwrite(path_to_save, image)


    def detect_main_people(self,
                           image=None,
                           image_path=None,
                           path_to_save=None):
        """"""
        if image == None:
            try:
                image = cv2.imread(image_path)
            except:
                print('Please provided image data or image path')
                return

        rmbg_image = remove(image)
        cv2.imwrite(f'temp/output.jpg', rmbg_image)

        rmbg_image = cv2.imread('temp/output.jpg')

        class_ids, scores, boxes, indicies = self.detect_objects(image=image,
                                   person_only=True)

        rmbg_class_ids, rmbg_scores, rmbg_boxes, rmgb_indices = self.detect_objects(image=rmbg_image,
                                   person_only=True)

        image_width = image.shape[1]
        border_width = int(image_width/100)

        if len(rmgb_indices) == 0:
            areas = [b[-2] * b[-1] for b in boxes]
            argmax_area = np.argmax(areas)

            box = boxes[argmax_area]

            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            self._draw_prediction(image,
                                  class_ids[argmax_area],
                                  scores[argmax_area],
                                  round(x),
                                  round(y),
                                  round(x+w),
                                  round(y+h),
                                  border_width,
                                  False)

            cv2.imwrite(path_to_save, image)

        else:
            for i, _ in enumerate(rmgb_indices):
                try:
                    box = rmbg_boxes[i]
                except:
                    i = i[0]
                    box = rmbg_boxes[i]

                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                self._draw_prediction(image,
                                      rmbg_class_ids[i],
                                      rmbg_scores[i],
                                      round(x),
                                      round(y),
                                      round(x+w),
                                      round(y+h),
                                      border_width,
                                      False)

            cv2.imwrite(path_to_save, image)


    def remove_ppl_in_background(self,
                                 image_path,
                                 path_to_save):
        """"""
        image = read_image(image_path)
        image_yolo = prepare_image_yolo(image)
        output_yolo = self.yolo(image_yolo)
        output_yolo = cut_result(output_yolo)
        final_image = create_new_image(image, output_yolo, self.p2p, [0, 24, 26])
        plt.imsave('temp/temp.jpg', final_image * 0.5 + 0.5)

        image = cv2.imread(image_path)
        rmbg_image = remove(image)
        cv2.imwrite(f'temp/rmgb_temp.jpg', rmbg_image)

        rmbg_image = cv2.imread('temp/rmgb_temp.jpg')
        final_image = cv2.imread('temp/temp.jpg')

        for row in range(final_image.shape[1]):
            for col in range(final_image.shape[0]):
                if sum(rmbg_image[col, row, :]) != 0:
                    final_image[col, row, :] = rmbg_image[col, row, :]

        cv2.imwrite(path_to_save, final_image)
