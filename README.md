# DSMtric
### 介绍
基于flask的数据集评测后端。（前端：咱负责设计，别的同学来做）。
延时评测：接收评测请求后，返回评测是否正常开始，然后后台进行评测。接收答案请求后，如果评测结束，则返回评测结果；否则，返回还没完成。

### 如果新加入一个指标
1、在service文件夹下对应文件内：
  a.写指标计算函数。
    1）函数输入为config文件夹下的Data类实例，所需数据集的数据或元数据是Data类实例的属性（根据需要在Data类里添加）。
    2）函数输出为0～1的值
  b.写指标触发函数。
    1）函数输入为config文件夹下的Data类实例，所需数据集的数据或元数据是Data类实例的属性（根据需要在Data类里添加）。
    2）函数功能包括：判断指标计算所需数据是否都提供，判断指标计算所需数据满足要求、（尽量）不会让指标计算函数报错
    3）返回布尔值
  c.如果需要使用模型
    1）在model文件夹下新建一个文件，将模型封装为一个类；在models.py里import这个类
    2）在MSystem.py里实例化这个类，并且加一个这个类的api接口。
    3）在tools文件夹下askmodel.py里写api接口的调用函数，通过调用这个函数使用模型。
  d.在最下面的列表中加入新的子列表，元素依次为指标名称、触发函数、计算函数

### 结构

### 工作流程
#### 评测请求
1、System.py接收评测请求后检查参数无误，后台启动ServiceMain.py进行数据集评测。

2、ServiceMain.py调用service文件夹内的函数进行数据集评测，完成后保存到data文件夹下对应位置。
#### 答案请求
1、System.py接收答案请求后，基于对应结果文件是否存在判断评测是否完成，并返回结果。
