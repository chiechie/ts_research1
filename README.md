# ts_research1
## 目录结构
1. ts_research1/doc/...: 相关文档
2. ts_research1/scripts/...: 建模脚本
3. ts_research1/utils/...: 工具函数

## 使用说明
1. 拉取文件
```bash
git clone https://github.com/chiechie/ts_research1.git
cd ./ts_research1/
```

2. 在setting.py中修改LOCAL_UDF_DATA_DIR变量，
设置自己的数据存放目录。

3. 
```bash
export PYTHONPATH$pwd
python scripts/step2_makeXY.py
python scripts/step3_model.py
```
