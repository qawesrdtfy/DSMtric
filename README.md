# DSMtric
### 介绍
基于flask的数据集评测后端，前端咱负责设计，别的同学来做。
### 延时评测
延时评测：接收评测请求后，返回评测是否正常开始，然后后台进行评测。接收答案请求后，如果评测结束，则返回评测结果；否则，返回还没完成。
### 结构
- System.py：后端控制器层，含有两个接口：评测请求、答案请求。
- ServiceMain.py：评测执行代码，是执行评测时后台运行的文件。含有参数解析、指标激活、指标计算（包括总分）、结果保存与完成标识四项功能。
- service：指标计算器文件夹
  - consistency.py：一致性
  - diversity.py：多样性
  - normative.py：规范性
  - trig_consistency.py：一致性函数对应的触发函数
  - trig_diversity.py：多样性函数对应的触发函数
  - trig_normative.py：规范性函数对应的触发函数
- tools：指标计算器的共用工具文件夹
- models：存放模型实体的文件夹
- data：存放数据集和结果的文件夹
  - dataset：数据集文件夹
  - result指标结果文件夹
