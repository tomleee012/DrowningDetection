import time
import cv2
import random
import numpy as np
import math
import torch
import json
import random

class PartsGrouping:
    def __init__(self):
        print('init')

    def __iou(self, boxA, boxB):  #(x,y,w,h)
        # determine the (x, y)-coordinates of the intersection rectangle
        if(boxA is None or boxB is None):
            return 0.0

        else:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xA2, yA2 = boxA[0] + boxA[2],  boxA[1] + boxA[3]
            xB2, yB2 = boxB[0] + boxB[2],  boxB[1] + boxB[3]

            xB = min(xA2, xB2)
            yB = min(yA2, yB2)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]

            viou = interArea / float(boxAArea + boxBArea - interArea)

            return viou

    def link_body_parts(self, bboxes, labelNames):
        body, head, ubody, lbody, id_list = [], [], [], [], []
        for id, box in enumerate(bboxes):
            labelname = labelNames[id]
            if labelname == 'body':
                body.append(box)
                id_list.append(id)
            elif labelname == 'head': head.append(box)
            elif labelname == 'body_upper': ubody.append(box)
            elif labelname == 'body_lower': lbody.append(box)

        # update parts boxes to each body
        parts_list = []
        link_info = {}

        for bid, b_box in enumerate(body):
            parts_list = []
            lik_info = {}

            #head
            max_iou, max_id = 0, None
            for id2, part_box in enumerate(head):
                viou = self.__iou(part_box, b_box)
                if viou > max_iou:
                    max_iou = viou
                    max_id = id2

            if max_id is not None:
                parts_list.append(head[max_id])
            else:
                parts_list.append([])

            #upper body
            max_iou, max_id = 0, None
            for id2, part_box in enumerate(ubody):
                viou = self.__iou(part_box, b_box)
                if viou > max_iou:
                    max_iou = viou
                    max_id = id2

            if max_id is not None:
                parts_list.append(ubody[max_id])
            else:
                parts_list.append([])

            #lower body
            max_iou, max_id = 0, None
            for id2, part_box in enumerate(lbody):
                viou = self.__iou(part_box, b_box)
                if viou > max_iou:
                    max_iou = viou
                    max_id = id2

            if max_id is not None:
                parts_list.append(lbody[max_id])
            else:
                parts_list.append([])


            #body
            parts_list.append(b_box)

            link_info.update( { bid:parts_list } )

        return link_info
