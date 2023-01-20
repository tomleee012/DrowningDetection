import cv2
import math

class POSE:
    def __init__(self, p_font_size=1.2, line_border=2):
        self.p_font_size = p_font_size
        self.line_border = line_border

    def __get_angle(self, a, b, c):
        ang = abs(round(math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])), 2))
        return  round((abs(ang)%180),2)

    def hv_pose(self, img, head, body, ubody, lbody, mark_draw=False):
        c_head, c_body, c_ubody, c_lbody = None, None, None, None

        if head is not None:
            if len(head)>0 :
                c_head = ( int(head[0] + head[2]/2), int(head[1] + head[3]/2) )

        if body is not None:
            if len(body)>0:
                c_body = ( int(body[0] + body[2]/2), int(body[1] + body[3]/2) )

        if ubody is not None:
            if len(ubody)>0:
                c_ubody = ( int(ubody[0] + ubody[2]/2), int(ubody[1] + ubody[3]/2) )

        if lbody is not None:
            if len(lbody)>0:
                c_lbody = ( int(lbody[0] + lbody[2]/2), int(lbody[1] + lbody[3]/2) )

        angel = 0
        v_pose=None
        #1. upper body, lower body
        if c_ubody is not None and c_lbody is not None:
            bottom_veretical_point = (9999, c_lbody[1])
            p1, p2, p3 = c_ubody, c_lbody, bottom_veretical_point
        # 2. head, upper body
        elif c_head is not None and c_ubody is not None:
            bottom_veretical_point = (9999, c_ubody[1])
            p1, p2, p3 = c_head, c_ubody, bottom_veretical_point
        #3. upper body, body
        elif c_ubody is not None and c_body is not None:
            bottom_veretical_point = (9999, c_body[1])
            p1, p2, p3 = c_ubody, c_body, bottom_veretical_point
        #4. lower body, body
        elif c_lbody is not None and c_body is not None:
            bottom_veretical_point = (9999, c_lbody[1])
            p1, p2, p3 = c_body, c_lbody, bottom_veretical_point
        #5. head, body
        elif c_head is not None and c_body is not None:
            bottom_veretical_point = (9999, c_body[1])
            p1, p2, p3 = c_head, c_body, bottom_veretical_point
        #head, lower body
        elif c_head is not None and c_lbody is not None:
            bottom_veretical_point = (9999, c_lbody[1])
            p1, p2, p3 = c_head, c_lbody, bottom_veretical_point
        else:
            angel = None

        #draw points
        if mark_draw is True:
            if c_head is not None: image = cv2.circle(img, c_head, 8, (0,0,255), 3)
            if c_body is not None: image = cv2.circle(img, c_body, 8, (0,0,255), 3)
            if c_ubody is not None: image = cv2.circle(img, c_ubody, 8, (0,255,0), 3)
            if c_lbody is not None: image = cv2.circle(img, c_lbody, 8, (0,255,0), 3)


        if angel == 0:
            angel = int(self.__get_angle(p1, p2, p3))
            #print('angel', angel)
            if mark_draw is True:
                image = cv2.line(image, p2, p3, (255, 0, 0), self.line_border)
                image = cv2.line(image, p1, p2, (0, 255, 0), self.line_border+1)

            if(angel <90) :
                ans_angel = angel
            else:
                ans_angel = 180 -angel

            if ans_angel>70 :
                pose_now = 'vertical'
                pose_color = (0,255,255)
                v_pose = 1
            else:
                pose_now = 'hotizontal'
                pose_color = (255,255,0)
                v_pose = 0

            pose_body = '{}({}-->{})'.format(pose_now, angel, angel)
            if mark_draw is True:
                cv2.putText(img, pose_body, (p2[0]+20, p3[1]-25), cv2.FONT_HERSHEY_SIMPLEX, self.p_font_size-0.4, pose_color, self.line_border)

        return v_pose, img
