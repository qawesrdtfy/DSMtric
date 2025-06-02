from tools.readata import *


modal2func = {
    "结构化数据": read_structure,
    "文本": read_str,
    "类别": read_str,
    "图像": read_picture,
    "图像目标": read_picture,
    "音频": read_audio,
    "语音": read_audio,
    "视频": read_video
}


class Data:
    def __init__(self, data: dict, dataset_dir: str) -> None:
        # X的模态：结构化数据、文本、图像、音频、视频、语音
        self.X_modal = data.get('X_modal', [])
        if not isinstance(self.X_modal,list):
            self.X_modal=[self.X_modal]
        # Y的模态：类别、文本、图像、音频、视频、语音、图像目标
        self.Y_modal = data.get('Y_modal', [])
        if not isinstance(self.Y_modal,list):
            self.Y_modal=[self.Y_modal]
        # 标注规则
        self.rule = self._get_rule(dataset_dir)
        # 每个样本的主题
        self.X_topic = self._get_X_topic(dataset_dir)
        # 每个样本每个标注员的标注结果
        self.Y_per_annotater = self._get_Y_per_annotater(dataset_dir)
        # X和Y内容
        self.X, self.Y = self._read_XY(dataset_dir)

    def _read_XY(self, dataset_dir):
        X_path = os.path.join(dataset_dir, 'X')
        X = {}
        for modal in self.X_modal:
            X[modal] = modal2func[modal](os.path.join(X_path, modal))
        for modal in ['图像', '音频', '语音', '视频']:
            if modal in self.X_modal:
                X[modal+'地址'] = [os.path.join(X_path, modal, one)
                                 for one in os.listdir(os.path.join(X_path, modal))]
                X[modal+'地址'].sort()

        Y_path = os.path.join(dataset_dir, 'Y')
        Y = {}
        for modal in self.Y_modal:
            Y[modal] = modal2func[modal](os.path.join(Y_path, modal))
        for modal in ['图像', '音频', '语音', '视频']:
            if modal in self.Y_modal:
                Y[modal+'地址'] = [os.path.join(Y_path, modal, one)
                                 for one in os.listdir(os.path.join(Y_path, modal))]
                X[modal+'地址'].sort()
        return X, Y

    def _get_Y_per_annotater(self, dataset_dir):
        Yp_path = os.path.join(dataset_dir, 'Y_per_annotater')
        if not os.path.exists(Yp_path):
            Y_per_annotater = {}
            for modal in self.Y_modal:
                Y_per_annotater[modal] = []
            return Y_per_annotater
        Yp_dirs = [os.path.join(Yp_path, one) for one in os.listdir(Yp_path)]
        Y_per_annotater = {}
        for modal in self.Y_modal:
            Y_per_modal = [modal2func[modal](os.path.join(Yp_dir, modal))
                           for Yp_dir in Yp_dirs]  # 标注员数，样本数
            Y_per_modal = list(zip(*Y_per_modal))  # 样本数，标注员数
            Y_per_annotater[modal] = Y_per_modal
        for modal in ['图像']:
            if modal in self.Y_modal:
                Y_per_modal = [sorted([os.path.join(Yp_dir, modal, file) for file in os.listdir(os.path.join(Yp_dir, modal))])
                               for Yp_dir in Yp_dirs]  # 标注员数，样本数
                Y_per_modal = list(zip(*Y_per_modal))  # 样本数，标注员数
                Y_per_annotater[modal+'地址'] = Y_per_modal
        return Y_per_annotater

    def _get_X_topic(self, dataset_dir):
        X_topic_file = os.path.join(dataset_dir, 'X_topic/topic.txt')
        if not os.path.exists(X_topic_file):
            return []
        with open(X_topic_file, 'r', encoding='utf-8') as f:
            topics = [one.strip('\n') for one in f.readlines()]
        return topics

    def _get_rule(self, dataset_dir):
        rule_file = os.path.join(dataset_dir, 'rule.txt')
        if not os.path.exists(rule_file):
            return ''
        with open(rule_file, 'r', encoding='utf-8') as f:
            rule = ' '.join([one for one in f.readlines()])
        return rule
