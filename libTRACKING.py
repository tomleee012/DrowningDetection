# How to use
#-----------------------------------------------
# from libObjTracking import OBJTRACK
# TRACK = OBJTRACK()
# TRACK.start(frameID=frameid, bboxes=yolo.bbox)  #first image
# tracking_info = TRACK.tracking(frameID=frameid, bboxes=yolo.bbox, th_iou=0.35, th_remove_ob=th_remove_ob)  #next image

import cv2
import numpy as np
from scipy import spatial
from PIL import ImageFont, ImageDraw, Image
from libPOSE import POSE
from libGROUPING import PartsGrouping

class OBJTRACK:
    def __init__(self, p_font_size=1.2, line_border=2):
        self.th_iou = 0.9
        self.obj_info = {}
        #self.inTracking = False
        self.p_font_size = p_font_size
        self.line_border = line_border
        self.output = None

        self.poseBODY = POSE(p_font_size=p_font_size, line_border=line_border)
        self.GROUPING = PartsGrouping()

    def printText(self, bg, txt, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
        (b,g,r,a) = color

        if(type=="English"):
            cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, self.p_font_size,  (b,g,r), self.line_border, cv2.LINE_AA)

        else:
            ## Use simsum.ttf to write Chinese.
            fontpath = "fonts/wt009.ttf"
            font = ImageFont.truetype(fontpath, int(size*10*4))
            img_pil = Image.fromarray(bg)
            draw = ImageDraw.Draw(img_pil)
            draw.text(pos,  txt, font = font, fill = (b, g, r, a))
            bg = np.array(img_pil)

        return bg

    def iou(self, boxA, boxB):  #(x,y,w,h)
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

            iou = interArea / float(boxAArea + boxBArea - interArea)

            return iou

    def tracking(self, img, frameID, th_iou, th_remove_ob, yolo_bboxes, yolo_classes, print_id=False, draw_marks=False):
        obj_info = self.obj_info
        self.th_iou = th_iou
        obj_info_new, min_idname, min_box = {}, None, None
        bboxes = self.GROUPING.link_body_parts(yolo_bboxes, yolo_classes)

        remove_item = []
        for obj_name in obj_info:
            [init_box, box_last, box_now, count_lost, (head,box,lbody,ubody), pose_info, iou_overlap] = obj_info[obj_name]
            count_lost += 1
            if(count_lost>=th_remove_ob):
                remove_item.append(obj_name)

            obj_info.update( {obj_name: [init_box, box_last, box_now, count_lost, (head,box,lbody,ubody), pose_info, iou_overlap] } )

        for obj_name in remove_item:
            del obj_info[obj_name]

        #if not len(obj_info)>0:
        #    self.inTracking = False
        #    return obj_info, img

        for pid in bboxes:
            head = bboxes[pid][0]
            ubody = bboxes[pid][1]
            lbody = bboxes[pid][2]
            box = bboxes[pid][3] #body box

        #for oid, box in enumerate(bboxes):
            min_dist, min_idname = 1.0, None
            min_init_box, min_last_box = None, None
            for obj_name in obj_info:
                [box_init, box_last, box_now, count_lost, (tmp_head,tmp_box,tmp_lbody,tmp_ubody), pose_info, iou_overlap] = obj_info[obj_name]
                v_iou = self.iou(box, box_now)

                dist = abs(1.0-abs(v_iou))
                if(min_dist>dist and dist<=th_iou):
                    min_iou = v_iou
                    min_dist = dist
                    min_idname = obj_name
                    min_init_box = box_init
                    min_last_box = box_now


            if print_id is False:
                pose_info, _ = self.poseBODY.hv_pose(img.copy(), head, box, ubody, lbody, mark_draw=draw_marks)
            else:
                pose_info, img = self.poseBODY.hv_pose(img, head, box, ubody, lbody, mark_draw=draw_marks)


            if(min_idname is not None):
                obj_info.update( { min_idname:[min_init_box, min_last_box, box, 0, (head,box,lbody,ubody), pose_info, min_iou] }  )

            else:
                obj_name = "{}".format(str(frameID).zfill(6))
                lastbox = box
                obj_info.update( { obj_name:[box, None, box, 0, (head,box,lbody,ubody), pose_info, None] }  )

        if print_id is True and len(obj_info)>0:
            for obj_name in obj_info:
                [x,y,w,h] = obj_info[obj_name][2]
                # if y<0: y=0
                # if obj_info[obj_name][3]==0: cv2.putText(img,  obj_name, (int(x+w/2),y+50), cv2.FONT_HERSHEY_SIMPLEX, \
                #     self.p_font_size,  (255,0,255), self.line_border, cv2.LINE_AA)


        self.obj_info = obj_info

        data_update = self.get_data()
        self.IDs = data_update[0]
        self.alignment = data_update[2]
        self.classesseen = data_update[3]
        self.ioudict = data_update[4]
        self.bboxlefttop = data_update[5]
        self.bboxrightbottom = data_update[6]

        return obj_info, img

    def get_data(self):
        obj_info = self.obj_info
        name_ids, bodys, poses, body_parts, ious = [], [], [], [], []
        lefttop, rightbottom = [], []
        for nid in obj_info:
            name_ids.append(nid)
            (head, body, lbody, ubody) = obj_info[nid][4]
            bodys.append(body)

            parts = []
            for id, bpart in enumerate(obj_info[nid][4]):
                if bpart is not None:
                    parts.append(id)

            body_parts.append(parts)
            poses.append(obj_info[nid][5])
            ious.append(obj_info[nid][6])
            lefttop.append((body[0],body[1]))
            rightbottom.append((body[0]+body[2], body[1]+body[3]))

        return [name_ids, bodys, poses, body_parts, ious, lefttop, rightbottom]
