from ..tools.readata import *


modal2func={
    "结构化数据":read_structure,
    "文本":read_str,
    "类别":read_str,
    "图像":read_picture,
    "音频":read_audio,
    "语音":read_audio,
    "视频":read_video
}

class Data:
    def __init__(self, data:dict, dataset_dir:str) -> None:
        # X的模态：结构化数据、文本、图像、音频、视频、语音
        self.X_modal=data.get('X_modal',[])
        # Y的模态：类别、文本、图像、音频、视频、语音
        self.Y_modal=data.get('Y_modal',[])
        # 每个样本每个标注员的标注结果
        self.Y_per_annotater=data.get('Y_per_annotater',[])
        # 每个样本的主题
        self.X_topic=data.get('X_topic',[])
        # 如果Y模态含有图像，则有此属性：每个图像的路径
        self.Y_pic_paths=self._get_Y_pic_paths(dataset_dir)
        # X和Y内容
        self.X,self.Y=self._read_XY(dataset_dir)
    
    def _read_XY(self, dataset_dir):
        X_path=os.path.join(dataset_dir,'X')
        X={}
        for modal in self.X_modal:
            X[modal]=modal2func[modal](os.path.join(X_path,modal))

        Y_path=os.path.join(dataset_dir,'Y')
        Y={}
        for modal in self.Y_modal:
            Y[modal]=modal2func[modal](os.path.join(Y_path,modal))
        return X,Y
    
    def _get_Y_pic_paths(self, dataset_dir):
        if '图像' not in self.Y_modal:
            return []
        Y_pic_dir=os.path.join(dataset_dir,'Y','图像')
        return [os.path.join(Y_pic_dir,one) for one in os.listdir(Y_pic_dir)]
