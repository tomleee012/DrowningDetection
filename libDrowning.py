import statistics
import cv2
import time

# DROWNING.now_actions / DROWNING.last_actions for current & last frames
class DROWNING:
    def __init__(self, moving_avg = 3, counter_type=0 ):
        self.moving_avg = moving_avg
        self.now = []
        self.last = []
        self.now_actions = {}
        self.drowning_hotlist = {}
        self.predrowning_hotlist = {}
        self.counter_type = counter_type

    def vh_move_ratio(self, p1, p2):
        x_move = abs(p2[0] - p1[0])
        y_move = abs(p2[1] - p1[1])

        ratio_vh = y_move / x_move

        return ratio_vh

    def punch(self, pdata):
        now = self.now.copy()
        last = self.last.copy()

        now.append(pdata.copy())
        if len(now)>self.moving_avg:
            first = now.pop(0)
            last.append(first)

        if len(last)>self.moving_avg:
            last.pop(0)

        self.now = now.copy()
        self.last = last.copy()

        #set moving list for last datas
        last_moving_list = {}
        for item_id, ldata in enumerate(self.last):

            for ID in ldata:
                if ID in last_moving_list:
                    (poses, heads, bodys, lbodys, ubodys, ious) = last_moving_list[ID]
                else:
                    poses, heads, bodys, lbodys, ubodys, ious = [], [], [], [], [], []

                poses.append(ldata[ID][5])
                (head,body,lbody,ubody) = ldata[ID][4]
                heads.append(head)
                bodys.append(body)
                lbodys.append(lbody)
                ubodys.append(ubody)
                ious.append(ldata[ID][6])

                last_moving_list.update( { ID:(poses, heads, bodys, lbodys, ubodys, ious) } )

        self.movelist_last = last_moving_list
        #print(' test punch', last_moving_list)

        #set moving list for now datas
        now_moving_list = {}
        for item_id, ndata in enumerate(self.now):

            for ID in ndata:
                if ID in now_moving_list:
                    (poses, heads, bodys, lbodys, ubodys, ious) = now_moving_list[ID]
                else:
                    poses, heads, bodys, lbodys, ubodys, ious = [], [], [], [], [], []

                poses.append(ndata[ID][5])
                (head,body,lbody,ubody) = ndata[ID][4]
                heads.append(head)
                bodys.append(body)
                lbodys.append(lbody)
                ubodys.append(ubody)
                ious.append(ndata[ID][6])

                now_moving_list.update( { ID:(poses, heads, bodys, lbodys, ubodys, ious) } )

        #print('now', now)
        #print('last', last)
        #print('-----------------------------------------------------------')
        self.movelist_last = last_moving_list
        self.movelist_now = now_moving_list
        #print(' test punch', now_moving_list)
        self.moving_summarize()

    def avg_boxes(self, boxes):
        counts = len(boxes)
        if not counts>0:
            return None

        i = 0
        xx,yy,ww,hh = 0, 0, 0, 0
        for box in boxes:
            if not len(box)>0: continue

            (x,y,w,h) = box
            xx += x
            yy += y
            ww += w
            hh += h
            i += 1

        if i>0:
            return (int(xx/i), int(yy/i), int(ww/i), int(hh/i))
        else:
            return None

    def moving_summarize(self):
        last_movelist = self.movelist_last
        now_movelist = self.movelist_now

        #print('last_movelist', last_movelist)
        #print('now_movelist', now_movelist)

        last_summarize = {}
        for ID in last_movelist:
            poses = last_movelist[ID][0]
            heads = last_movelist[ID][1]
            bodys = last_movelist[ID][2]
            lbodys = last_movelist[ID][3]
            ubodys = last_movelist[ID][4]
            ious = last_movelist[ID][5]

            last_pose = max(poses, key=poses.count)   #select the max occurrences of pose from last
            #remove none for ious
            ious = [i for i in ious if i]
            if len(ious)>0:
                last_iou = statistics.mean(ious)
            else:
                last_iou = None

            last_head = self.avg_boxes(heads)
            last_body = self.avg_boxes(bodys)
            last_lbody = self.avg_boxes(lbodys)
            last_ubody = self.avg_boxes(ubodys)

            last_summarize.update( { ID: [last_pose, last_head, last_body, last_lbody, last_ubody, last_iou]} )

        self.last_actions = last_summarize

        now_summarize = {}
        for ID in now_movelist:
            poses = now_movelist[ID][0]
            heads = now_movelist[ID][1]
            bodys = now_movelist[ID][2]
            lbodys = now_movelist[ID][3]
            ubodys = now_movelist[ID][4]
            ious = now_movelist[ID][5]

            now_pose = max(poses, key=poses.count)   #select the max occurrences of pose from last
            #remove none in ious
            ious = [i for i in ious if i]
            if len(ious)>0:
                now_iou = statistics.mean(ious)
            else:
                now_iou = None

            now_head = self.avg_boxes(heads)
            now_body = self.avg_boxes(bodys)
            now_lbody = self.avg_boxes(lbodys)
            now_ubody = self.avg_boxes(ubodys)

            now_summarize.update( { ID: [now_pose, now_head, now_body, now_lbody, now_ubody, now_iou]} )

        self.now_actions = now_summarize
        #print(now_summarize)

    def detect_drowning(self, img, th_hot_list, drown_sure, poses_required):
        now_actions = self.now_actions
        last_actions = self.last_actions
        hotlist = self.drowning_hotlist

        for ID in now_actions:
            now_pose = now_actions[ID][0]
            now_head = now_actions[ID][1]
            now_body = now_actions[ID][2]
            now_lbody = now_actions[ID][3]
            now_ubody = now_actions[ID][4]
            now_iou = now_actions[ID][5]

            now_time = time.time()
            if ID in hotlist:
                if self.counter_type == 0:
                    counts = now_time - hotlist[ID][1]
                else:
                    counts = hotlist[ID][0]

                if counts > drown_sure:
                    bcolor = (0,0,255)
                    cv2.putText(img,  'drowning!', (now_body[0],now_body[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.85,  bcolor, 2, cv2.LINE_AA)

            if ID in last_actions:
                last_pose = last_actions[ID][0]
                last_head = last_actions[ID][1]
                last_body = last_actions[ID][2]
                last_lbody = last_actions[ID][3]
                last_ubody = last_actions[ID][4]
                last_iou = last_actions[ID][5]

                now_body_centroid = ( now_body[0]+int(now_body[2]/2) , now_body[1]+int(now_body[3]/2) )
                last_body_centroid = ( last_body[0]+int(last_body[2]/2) , last_body[1]+int(last_body[3]/2) )

                #print('now_body_centroid, last_body_centroid', (now_body_centroid,last_body_centroid))
                y_movement = now_body_centroid[1] - last_body_centroid[1]
                x_movement = now_body_centroid[0] - last_body_centroid[0]
                ratio_ymove = (abs(y_movement) / now_body[3]) * 100
                ratio_xmove = (abs(x_movement) / now_body[2]) * 100

                #print(ID, 'Drowning move ratio', ratio_xmove+ratio_ymove)

                if (now_pose == poses_required or poses_required==2) and (ratio_xmove+ratio_ymove)<th_hot_list:  #register or update the count
                    if ID in hotlist:
                        counts = hotlist[ID][0] + 1
                        start_timer = hotlist[ID][1]
                    else:
                        counts = 0
                        start_timer = now_time

                    yn_drown = False
                    if self.counter_type == 1:  #Drowning counter by using frames
                        if counts>drown_sure:
                            yn_drown = True

                    else:  # Drowning counter is timer
                        if time.time() - start_timer > drown_sure:
                            yn_drown = True

                    if yn_drown is True:
                        bcolor = (0,0,255)
                        cv2.putText(img,  'drowning!', (now_body[0],now_body[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.85,  bcolor, 2, cv2.LINE_AA)
                    else:
                        bcolor = (0,255,255)

                    cv2.rectangle(img, (now_body[0],now_body[1]), (now_body[0]+now_body[2], now_body[1]+now_body[3]), bcolor, 1)
                    hotlist.update( { ID:[counts, start_timer, now_body] })

                elif ID in hotlist:
                    counts = hotlist[ID][0] + 1
                    start_timer = hotlist[ID][1]

                    if self.counter_type == 1:
                        if counts<drown_sure:
                            hotlist.pop(ID)
                    else:
                        if now_time - start_timer < drown_sure:
                            hotlist.pop(ID)

                self.drowning_hotlist = hotlist

        return img

    def detect_predrowning(self, img, th_hot_list, predrown_sure, poses_required):
        now_actions = self.now_actions
        last_actions = self.last_actions

        hotlist = self.predrowning_hotlist
        #print('last', last_actions)
        #print('now', now_actions)

        for ID in now_actions:
            now_pose = now_actions[ID][0]
            now_head = now_actions[ID][1]
            now_body = now_actions[ID][2]
            now_lbody = now_actions[ID][3]
            now_ubody = now_actions[ID][4]
            now_iou = now_actions[ID][5]

            now_time = time.time()
            if ID in hotlist:
                if self.counter_type == 0:
                    counts = now_time - hotlist[ID][1]
                else:
                    counts = hotlist[ID][0]
                '''
                if counts> predrown_sure:
                    bcolor = (0,0,255)
                    cv2.putText(img,  'pre-drowning!', (now_body[0],now_body[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.85,  bcolor, 2, cv2.LINE_AA)
                '''
            if ID in last_actions:
                last_pose = last_actions[ID][0]
                last_head = last_actions[ID][1]
                last_body = last_actions[ID][2]
                last_lbody = last_actions[ID][3]
                last_ubody = last_actions[ID][4]
                last_iou = last_actions[ID][5]

                now_body_centroid = ( now_body[0]+int(now_body[2]/2) , now_body[1]+int(now_body[3]/2) )
                last_body_centroid = ( last_body[0]+int(last_body[2]/2) , last_body[1]+int(last_body[3]/2) )

                y_movement = abs( (now_body_centroid[1] - last_body_centroid[1]) / last_body[2])
                x_movement = abs( (now_body_centroid[0] - last_body_centroid[0]) / last_body[3])
                #if x_movement == 0: x_movement = 1.0

                ratio_move = y_movement - x_movement
                if (ratio_move>-0.05 and ratio_move<0.05):
                    ratio_move = th_hot_list + 1.0


                #print(ID, 'y_movement {}, x_movement {}, ratio_move {}, th_hot_list {}'.format(y_movement, x_movement, ratio_move, th_hot_list))

                if (now_pose == poses_required or poses_required==2) and ratio_move>th_hot_list:  #register or update the count
                    if ID in hotlist:
                        counts = hotlist[ID][0] + 1
                        start_timer = hotlist[ID][1]
                    else:
                        counts = 1
                        start_timer = now_time

                    yn_drown = False
                    if self.counter_type == 1:  #Drowning counter by using frames
                        if counts> predrown_sure:
                            yn_drown = True
                    else:  # Drowning counter is timer
                        if now_time - start_timer > predrown_sure:
                            yn_drown = True

                    if yn_drown is True:
                        bcolor = (0,0,255)
                        cv2.putText(img,  'pre-drowning!', (now_body[0],now_body[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.85,  bcolor, 2, cv2.LINE_AA)
                    else:
                        bcolor = (0,255,255)
                        cv2.putText(img,  'normal', (now_body[0],now_body[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.85,  bcolor, 2, cv2.LINE_AA)

                    cv2.rectangle(img, (now_body[0],now_body[1]), (now_body[0]+now_body[2], now_body[1]+now_body[3]), bcolor, 1)
                    hotlist.update( { ID:[counts, start_timer, now_body] })
                    #print('test -->', ID, counts, start_timer, now_time, ' = ', now_time-start_timer)

                elif ID in hotlist:
                    counts = hotlist[ID][0] + 1
                    start_timer = hotlist[ID][1]

                    if self.counter_type == 1:
                        if counts<predrown_sure:
                            hotlist.pop(ID)
                    else:
                        if now_time - start_timer < predrown_sure:
                            print('TEST pop it, ratio_move=',ratio_move) 
                            hotlist.pop(ID)

                self.predrowning_hotlist = hotlist

        return img
