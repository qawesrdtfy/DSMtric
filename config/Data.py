class Data:
    def __init__(self, data:dict) -> None:
        # X的模态：结构化数据、文本、图像、音频、视频、语音
        self.X_modal=data.get('X_modal',[])
        # Y的模态：类别、文本、图像、音频、视频、语音
        self.Y_modal=data.get('Y_modal',[])
        # 每个样本每个标注员的标注结果
        self.Y_per_annotater=data.get('Y_per_annotater',[])
        # 每个样本的主题
        self.X_topic=data.get('X_topic',[])
        # 如果Y模态含有图像，则有此属性：每个图像的路径
        # TODO 需要自己获得
        self.Y_pic_paths=data.get('Y_pic_paths',[])
    
    def set_X(self, X):
        self.X=X
    
    def set_Y(self, Y):
        self.Y=Y