import subprocess

class rtmpPipe:
    def __init__(self) -> None:
        pass
    def createPipe(self, width, height, fps, rtmp_url):
        command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "{}x{}".format(width, height),
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'flv',
            rtmp_url]

        # using subprocess and pipe to fetch frame data
        p = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.pipe=p
        return p
    def send(self,frame):
        self.pipe.stdin.write(frame.tobytes())
