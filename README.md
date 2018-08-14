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
设置自己的数据存放路径,如
'shihuanzhao': '/Users/stellazhao/tencent_workplace/labgit/dataming/ts_research1/test_data/',
该路径下目录结构默认如下设置：
```
|-- test_data
|   --labeled_data
|   --order_book_DB
|   --step2_model_data
```
3. 
```bash
export PYTHONPATH=$pwd
python scripts/step2_makeXY.py
python scripts/step3_model.py
```
