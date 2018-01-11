class Vehicle():
    def __init__(self):
        self.detected = False
        self.n_detections = 0
        self.n_nondetections = 0
        self.xpixels = None
        self.ypixels = None
        self.recent_xfitted = []
        self.bestx = None
        self.recent_yfitted = []
        self.besty = None
        self.recent_wfitted = []
        self.bestw = None
        self.recent_hfitted = []
        self.besth = None