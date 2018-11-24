> AI-Challenger 2018 天气预报赛道
> 
> 队伍名称：LoganLin

# 代码说明

## 环境配置

1. 编程语言: Python3.6
2. 机器学习框架：XGBoost
3. 调用的库：
> - numpy 1.15.1
> - pandas 0.23.4
> - netCDF4 1.4.1

## 代码文件结构说明

### Python文件

- `load_data.py`
> 读取原始数据，并转换为`pd.DataFrame`。
- `generate_set.py`
> 生成能送入`XGBoost`的训练数据与预测数据。
- `train_xgb.py`
> 训练`XGBoost`模型，并存入模型文件
- `xgb_predict.py`
> 使用训练完成的`XGBoost`模型进行预测，并生成提交文件。

### Jupyter Notebook

- `gen_train_data.ipynb`
> 生成指定时间段的训练数据。
- `eval_xgboost.ipynb`
> 网格化搜索`XGBoost`最佳参数。
- `train_xgboost.ipynb`
> 训练`XGBoost`模型。
- `xgb_predict.ipynb`
> 使用`XGBoost`模型进行预测。

## 训练代码的使用说明
训练模型前，首先需要生成训练数据。可使用`generate_set.combine_periods`函数。
> 函数的使用可参考`gen_train_data.ipynb`中的写法。

生成完毕后使用`pickle.dump()`函数将数据保存为文件。

训练新的数据集时，我会先用网格搜索计算模型的最佳参数。可使用`train_xgb.gen_best_parameters`函数。
> 参考`eval_xgboost.ipynb`中的写法。

读取计算得到的参数，使用`train_xgb.train_all_xgb`函数，即可训练一组模型。
> 参考`train_xgboost.ipynb`中的写法。

## 测试代码的使用说明
使用`generate_set.generate_x_oneday`函数可生成提交一次预测需要的输入特征数据。

随后调用`xgb_predict.xgb_predict_all`函数读取模型文件，生成预测结果。

最后调用`xgb_predict.construct_submission`函数可对预测结果进行规范化，并生成提交结果所需的文件。
> 参考`xgb_predict.ipynb`中的写法。