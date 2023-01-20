from libDNNYolo import opencvYOLO
from libTRACKING import OBJTRACK
from libDrowning import DROWNING
import cv2, imutils, argparse
from rtmp import rtmpPipe

argparser = argparse.ArgumentParser(
    description='Drowning Detection')

argparser.add_argument(
    '-s',
    '--source',
    default='videos/drowning1.mp4',
    type=str,
    help='Source of a video')

argparser.add_argument(
    '-g',
    '--gpu',
    default=True,
    type=bool,
    help='Whether GPU acceleration is available')

argparser.add_argument(
    '-w',
    '--write',
    default=True,
    type=bool,
    help='Whether to output a video')

argparser.add_argument(
    '-p',
    '--push',
    default=False,
    type=bool,
    help='Whether to push a video')

score = 0.45
nms = 0.1

# Parameters for the model, you don't have to change them
#===================================================================================

def ModelPara(args):
    path_objname = r"models/yolov5/obj.names"
    path_weights = r"models/yolov5/yolov5s_bodyparts.pt"

    yolomodel = opencvYOLO( \
        mtype='yolov5', imgsize=(640,640), \
        objnames=path_objname, \
        weights=path_weights, \
        darknetcfg='', score=score, nms=nms, gpu=args.gpu)

    TRACK = OBJTRACK(p_font_size=1.0, line_border=2)
    DROWN = DROWNING(moving_avg=3, counter_type=0) # counter_type: 0 - use seconds, 1 - use frames

    return yolomodel, TRACK, DROWN

#===================================================================================

if __name__ == "__main__":
    args = argparser.parse_args()
    yolomodel, TRACK, DROWN = ModelPara(args)

    print('Push Q to quit the program.')
    INPUT = cv2.VideoCapture(args.source)

    InputName = args.source.split('.')
    video_out = f'{InputName[0]}_detected.mp4'

    frameid = 0
    if video_out != "":
        width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

        if args.write:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_out,fourcc, 20, (int(width),int(height)))

    yolomodel.land_y = int(0 * height) # 0 percent of non-land area from top
    hasFrame, frame = INPUT.read()

    if args.push:
        rtmpUrl="rtmp://192.168.100.240/live/livestream31"
        pipe=rtmpPipe()
        fpsss = INPUT.get(cv2.CAP_PROP_FPS)
        frame_sizess = (width, height)
        pipe.createPipe(frame_sizess[0], frame_sizess[1], fpsss, rtmpUrl)

    while hasFrame:
        img = frame.copy()
        frameid += 1
        img = yolomodel.getObject(img, score, nms, drawBox=False)
        tracking_info, img = TRACK.tracking(img=img, frameID=frameid, \
            # th_remove_ob: object will be removed if disappeared for x frames
            # th_iou: iou below the number will see as same one
            th_iou=0.55, th_remove_ob=30, \
            yolo_bboxes=yolomodel.bbox, yolo_classes=yolomodel.labelNames, print_id=True, draw_marks=False )

        #print(tracking_info)
        #body_data = TRACK.get_data()
        DROWN.punch(tracking_info)
        # th_hot_list: distance threshold for movement(X+Y) is lower than th, between current & last frames
        # poses_required: pose when drowning, 0:horizontal, 1:vertical, 2:h and v
        # drown_sure: my GPU's FPS=60, video is 25 fps, so 25*10/60 = 4
        img = DROWN.detect_drowning(img, th_hot_list=18.0, drown_sure=4, poses_required=0)
        # th_hot_list: distance threshold for movement (X+Y) is higher than th, between current & last frames
        # poses_required: pose when drowning, 0:horizontal, 1:vertical, 2: both
        # predrown_sure: my GPU's FPS=60, video is 25 fps, so 25*10/60 = 4
        img = DROWN.detect_predrowning(img, th_hot_list=0, predrown_sure=10, poses_required=2)
        '''
        print('IDs', TRACK.IDs)
        print('alignment', TRACK.alignment)
        print('classes seen', TRACK.classesseen)
        print('IOU overlap', TRACK.ioudict)
        print('Body left-top', TRACK.bboxlefttop)
        print('Body right-bottom', TRACK.bboxrightbottom)
        '''
        cv2.imshow('test', imutils.resize(img, height=800))
        if args.push:
            pipe.send(frame)

        k = cv2.waitKey(1)
        if(k==113):
            break

        if args.write:
            out.write(img)

        hasFrame, frame = INPUT.read()

    if args.write:
        out.release()
