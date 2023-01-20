import time
import cv2
import random
import numpy as np
import math
import torch
import json
import random

class opencvYOLO:
    def __init__(self, mtype='darknet', imgsize=(416,416), objnames="coco.names", \
            weights="yolov3.weights", darknetcfg="yolov3.cfg", score=0.25, nms=0.6, gpu=False):
        self.mtype = mtype
        self.imgsize = imgsize
        self.score = score
        self.nms = nms

        self.land_y = 0.0

        self.inpWidth = self.imgsize[0]
        self.inpHeight = self.imgsize[1]
        self.classes = None
        with open(objnames, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
   
        if mtype == 'yolov5':
            dnn = torch.hub.load('ultralytics/yolov5', 'custom', weights, force_reload=False)
            dnn.conf = score
            dnn.iou = nms
        else:
            dnn = cv2.dnn.readNetFromDarknet(darknetcfg, weights)

            if gpu is True:
                dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        tcolors = []
        for id, cname in enumerate(self.classes):
            tcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            tcolors.append(tcolor)

        self.tcolors = tcolors

        self.net = dnn

    def setScore(self, score=0.5):
        if self.mtype == 'yolov5':
            self.net.conf = score
        else:
            self.score = score

    def setNMS(self, nms=0.8):
        if self.mtype == 'yolov5':
            self.net.iou = nms
        else:
            self.nms = nms

    def _bg_text(self, img, labeltxt, loc, txtdata):
        (x,y) = loc
        (font, font_scale, font_thickness, text_color, text_color_bg) = txtdata

        text_size, _ = cv2.getTextSize(labeltxt, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rx, ry = x, y - int(text_h*1.05)
        if ry<0: ry = 0

        rx2, ry2 = rx+int(text_w*1.05), ry+int(text_h*1.05)
        if ry<0: ry =0
        if rx2>img.shape[1]: rx2=img.shape[1]
        if ry2>img.shape[0]: ry2=img.shape[0]

        cv2.rectangle(img, (rx,ry), (rx2, ry2), text_color_bg, -1)
        img = self.printText(img, labeltxt, color=(text_color[0],text_color[1],text_color[2],0), size=font_scale, \
                pos=(x, y)) 

        return img

    def printText(self, bg, txt, color=(0,255,0,0), size=0.7, pos=(0,0)):
        (b,g,r,a) = color
        cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

        return bg

    def __getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def __postprocess(self, frame, outs, labelWant, drawBox, bold, textsize):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        tcolors = self.tcolors
 
        classIds = []
        labelName = []
        confidences = []
        boxes = []
        boxbold = []
        labelsize = []
        #boldcolor = []
        #textcolor = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                label = self.classes[classId]
                if( (labelWant=="" or (label in labelWant)) and (confidence > self.score) ):
                    #print(detection)
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append((left, top, width, height))
                    boxbold.append(bold)
                    labelName.append(label)
                    labelsize.append(textsize)
                    #boldcolor.append(bcolor)
                    #textcolor.append(tcolor)

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.score, self.nms)
        self.indices = indices

        nms_classIds = []
        #labelName = []
        nms_confidences = []
        nms_boxes = []
        nms_boxbold = []
        nms_labelNames = []
        body_boxes = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            if top<self.land_y: continue

            nms_confidences.append(confidences[i])
            nms_classIds.append(classIds[i])
            nms_boxes.append(boxes[i])
            nms_labelNames.append(labelName[i])

            if labelName[i] == 'body':
                body_boxes.append(boxes[i])

            if(drawBox==True):
                txt_color = tcolors[classIds[i]]

                frame = self.__drawPred(frame, classIds[i], confidences[i], boxbold[i], txt_color,
                    left, top, left + width, top + height)

        self.bbox = nms_boxes
        self.classIds = nms_classIds
        self.scores = nms_confidences
        self.labelNames = nms_labelNames
        self.frame = frame
        self.body_boxes = body_boxes

        return frame

    def __drawPred(self, frame, className, conf, bold, textcolor, left, top, right, bottom):
        if self.mtype == 'darknet':
            className = self.classes[int(className)]


        label = '{}({}%)'.format(className, int(conf*100))
        border_rect = 2
        if(frame.shape[0]<720): border_rect = 1

        textsize = (right - left) / 250.0
        txtbgColor = (255-textcolor[0], 255-textcolor[1], 255-textcolor[2])
        txtdata = (cv2.FONT_HERSHEY_SIMPLEX, textsize, border_rect, textcolor, txtbgColor)

        cv2.rectangle(frame, (left, top), (right, bottom), txtbgColor, border_rect)
        frame = self._bg_text(frame, label, (left+1, top+1), txtdata)
        return frame

    def getObject(self, frame, score, nms, labelWant='', drawBox=False):
        textsize = 0.8
        tcolors = self.tcolors
        bold = 1
        if frame.shape[0]>720 and frame.shape[1]>1024: bold = 2

        if self.mtype == 'yolov5':
            self.net.conf = score  # confidence threshold (0-1)
            self.net.iou = nms  # NMS IoU threshold (0-1)

            results = self.net(frame[...,::-1], size=self.imgsize[0])
            predictions = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

            bboxes, names, cids, scores = [], [], [], []
            for p in predictions:
                xmin, xmax, ymin, ymax = int(p['xmin']), int(p['xmax']), int(p['ymin']), int(p['ymax'])
                if ymin<self.land_y: continue

                bboxes.append((xmin,ymin,xmax-xmin,ymax-ymin))
                scores.append(float(p['confidence']))
                names.append(p['name'])
                cids.append(p['class'])

                if(drawBox==True):
                    txt_color = tcolors[p['class']]

                    frame = self.__drawPred(frame, p['name'], float(p['confidence']), bold, txt_color,
                        xmin, ymin, xmax, ymax, type=char_type)

            self.bbox = bboxes
            self.classIds = cids
            self.scores = scores
            self.labelNames = names


        else:
            net = self.net
            blob = cv2.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
            net.setInput(blob)
            # Runs the forward pass to get output of the output layers
            outs = net.forward(self.__getOutputsNames(net))
            # Remove the bounding boxes with low confidence
            frame = self.__postprocess(frame, outs, labelWant, drawBox, bold, textsize)
            self.objCounts = len(self.indices)
            # Put efficiency information. The function getPerfProfile returns the 
            # overall time for inference(t) and the timings for each of the layers(in layersTimes)
            #t, _ = net.getPerfProfile()

        self.frame = frame

        return frame
