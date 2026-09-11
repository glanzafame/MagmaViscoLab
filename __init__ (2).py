def __init__(self,sample=None,main_window=None):
    super().__init__()

    self.sample = sample
    self.main_window = main_window

    self.X = None
    self.Y = None
    self.Z = None

    self.setWindowTitle("MagmaViscoLab - 3D Plot")