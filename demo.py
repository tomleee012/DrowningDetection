from libDNNYolo import opencvYOLO
from libTRACKING import OBJTRACK
from libDrowning import DROWNING
from configparser import ConfigParser
import cv2
import imutils
from rtmp import rtmpPipe

cfg = ConfigParser()
cfg.read("drowning.ini",encoding="utf-8")

media = cfg.get("mediaSource", "media")

write_video = cfg.getboolean("recordVideo", "write_video")
framerate = cfg.getint("recordVideo", "framerate")
video_out = cfg.get("recordVideo", "video_out")

yolo_type = cfg.get("yoloDetect", "yolo_type")
score = cfg.getfloat("yoloDetect", "score")
nms = cfg.getfloat("yoloDetect", "nms")
dawrbox_frame = cfg.getboolean("yoloDetect", "drawbox_frame")
gpu = cfg.getboolean("yoloDetect", "gpu")

th_remove_ob = cfg.getint("tracking", "th_remove_ob")
th_iou_tracking = cfg.getfloat("tracking", "th_iou_tracking")

p_font_size = cfg.getfloat("indicatorDisplay", "p_font_size")
line_border = cfg.getint("indicatorDisplay", "line_border")

moving_avg = cfg.getint("drowningGlobal", "moving_frames_avg")
landline_under_y = cfg.getfloat("drowningGlobal", "landline_under_y")
counter_type = cfg.getint("drowningGlobal", "counter_type")

#Drowning detect
draw_marks = cfg.getboolean("drowningDetect", "draw_marks")
th_add_drownlist = cfg.getfloat("drowningDetect", "th_add_drownlist")
poses_drowning = cfg.getint("drowningDetect", "poses_drowning")
drown_sure_frames = cfg.getint("drowningDetect", "drown_sure_frames")
drown_sure_seconds = cfg.getint("drowningDetect", "drown_sure_seconds")

#Pre-drowing detect
predraw_marks = cfg.getboolean("predrowningDetect", "predraw_marks")
th_add_predrownlist = cfg.getfloat("predrowningDetect", "th_add_predrownlist")
poses_predrowning = cfg.getint("predrowningDetect", "poses_predrowning")
predrown_sure_frames = cfg.getint("predrowningDetect", "predrown_sure_frames")
predrown_sure_seconds = cfg.getint("predrowningDetect", "predrown_sure_seconds")

# Parameters for system, you don't have to change them
#===================================================================================

media = media.replace('\\', '/')
video_out = video_out.replace('\\', '/')

if yolo_type == 'yolov3':
    path_objname = r"models/yolov3/obj.names"
    path_weights = r"models/yolov3/yolov3_last.weights"
    path_darknetcfg = r"models/yolov3/yolov3.cfg" #yolov5 don't need this, keep it ''

    model_size = (608,608)
    yolomodel = opencvYOLO( \
        mtype='darknet', imgsize=model_size, \
        objnames=path_objname, \
        weights=path_weights, \
        darknetcfg=path_darknetcfg, \
        score=score, nms=nms, gpu=gpu)

else:
    path_objname = r"models/yolov5/obj.names"
    path_weights = r"models/yolov5/yolov5s_bodyparts.pt"
    path_darknetcfg = r"" #yolov5 don't need this, keep it ''

    model_size = (640,640)
    yolomodel = opencvYOLO( \
        mtype='yolov5', imgsize=model_size, \
        objnames=path_objname, \
        weights=path_weights, \
        darknetcfg='', score=score, nms=nms, gpu=gpu)


TRACK = OBJTRACK(p_font_size=p_font_size, line_border=line_border)
DROWN = DROWNING(moving_avg=moving_avg, counter_type=counter_type)
if counter_type == 0:
    drown_sure = drown_sure_seconds
    predrown_sure = predrown_sure_seconds
else:
    predrown_sure = predrown_sure_seconds
    prepredrown_sure = prepredrown_sure_seconds


#===================================================================================

if __name__ == "__main__":

    print('Push Q to quit the program.')

    INPUT = cv2.VideoCapture(media)

    frameid = 0
    if(video_out!=""):
        width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float


        if(write_video is True):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_out,fourcc, framerate, (int(width),int(height)))

    yolomodel.land_y = int(landline_under_y * height)
    hasFrame, frame = INPUT.read()

    rtmpUrl="rtmp://192.168.100.240/live/livestream31"
    pipe=rtmpPipe()
    fpsss = INPUT.get(cv2.CAP_PROP_FPS)
    frame_sizess = (width, height)
    pipe.createPipe(frame_sizess[0], frame_sizess[1], fpsss, rtmpUrl)

    while hasFrame:
        img = frame.copy()
        frameid += 1
        img = yolomodel.getObject(img, score, nms, drawBox=dawrbox_frame)
        tracking_info, img = TRACK.tracking(img=img, frameID=frameid, \
            th_iou=th_iou_tracking, th_remove_ob=th_remove_ob, \
            yolo_bboxes=yolomodel.bbox, yolo_classes=yolomodel.labelNames, print_id=True, draw_marks=draw_marks )

        #print(tracking_info)
        #body_data = TRACK.get_data()
        DROWN.punch(tracking_info)
        img = DROWN.detect_drowning(img, th_hot_list=th_add_drownlist, drown_sure=drown_sure, poses_required=poses_drowning)
        img = DROWN.detect_predrowning(img, th_hot_list=th_add_predrownlist, predrown_sure=predrown_sure, poses_required=poses_predrowning)
        '''
        print('IDs', TRACK.IDs)
        print('alignment', TRACK.alignment)
        print('classes seen', TRACK.classesseen)
        print('IOU overlap', TRACK.ioudict)
        print('Body left-top', TRACK.bboxlefttop)
        print('Body right-bottom', TRACK.bboxrightbottom)
        '''
        cv2.imshow('test', imutils.resize(img, height=800))
        # print(img.shape)
        pipe.send(frame)

        k = cv2.waitKey(1)
        if(k==113):
            break

        if write_video is True:
            out.write(img)

        hasFrame, frame = INPUT.read()

    if write_video is True: out.release()
