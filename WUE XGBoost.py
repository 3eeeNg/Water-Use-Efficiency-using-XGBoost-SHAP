import os
import time
import json
import warnings
import joblib

import numpy as np
import pandas as pd
import fastparquet

import xgboost as xgb
import shap
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 忽略显存相关的非致命警告
warnings.filterwarnings("ignore")

# ==========================================
# 全局配置与路径设置
# ==========================================
PARQUET_SET = [
    r"D:\WaterUseEfficiency\Arid\Arid_environmental_data.parquet",
    r"D:\WaterUseEfficiency\SemiArid\SemiArid_environmental_data.parquet",
    r"D:\WaterUseEfficiency\SubHumid\SubHumid_environmental_data.parquet",
    r"D:\WaterUseEfficiency\Humid\Humid_environmental_data.parquet"
]
CLA_NAMES = ["Arid", "SemiArid", "SubHumid", "Humid"]
VARIABLES = ['LAI', 'VPD', 'POP', 'PRE', 'SM', 'DSR', 'TEM']
OUT_DIR = r"D:\WaterUseEfficiency\WUECode\XGBoost-SHAP-Result"

# 超参数与运行配置
TARGET_SHAP_SAMPLES = 10000000  # SHAP 分析的目标采样数
N_TRAIN_YEARS_DEFAULT = 15        # 默认抽取的训练集年份数
CV_SPLITS = 5                     # 交叉验证的折数

# 设置 GPU 和 CPU 配置
GPU_CONFIG = {
    'tree_method': 'hist',  # 使用直方图算法，支持 GPU 加速
    'device': 'cuda:0'      # 指定使用第 0 块 GPU
}

# 确保主输出目录存在
os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================
# 核心功能函数
# ==========================================
def export_shap_data(shap_values: np.ndarray, X: pd.DataFrame, y: pd.Series, 
                     cla_name: str, output_path: str, base_values) -> None:
    """
    将 SHAP 值、原始特征和目标变量合并导出，并生成 SHAP 统计摘要。
    
    参数:
        shap_values: SHAP 计算结果矩阵
        X: 参与 SHAP 计算的特征数据
        y: 参与 SHAP 计算的目标变量真实值
        cla_name: 气候区类别名称 (用于文件命名)
        output_path: 结果保存目录
        base_values: SHAP 的基准值 (Expected value)
    """
    os.makedirs(output_path, exist_ok=True)

    # 1. 构造 SHAP DataFrame
    shap_cols = [f"SHAP_{col}" for col in X.columns]
    shap_df = pd.DataFrame(shap_values, columns=shap_cols, index=X.index)

    # 2. 安全地存储 base_values
    if np.isscalar(base_values):
        shap_df["base_values"] = base_values
    else:
        try:
            # 取多输出/多类别的第一个基准值，或直接广播数组
            shap_df["base_values"] = base_values[0] if isinstance(base_values, (list, np.ndarray)) else base_values
        except Exception as e:
            print(f"  [警告] 无法保存 base_values: {e}")

    # 3. 拼接并保存完整级联数据
    full_data = pd.concat([X, shap_df, y], axis=1)
    save_path = os.path.join(output_path, f"{cla_name}_shap_full_dataset.parquet")
    full_data.to_parquet(save_path, engine='fastparquet')
    
    # 4. 计算并保存特征重要性统计摘要
    summary_stats = pd.DataFrame({
        'feature': X.columns,
        'mean_shap': np.mean(shap_values, axis=0),
        'mean_abs_shap': np.mean(np.abs(shap_values), axis=0),  # 绝对值均值，反映整体重要性
        'std_shap': np.std(shap_values, axis=0)
    })
    summary_path = os.path.join(output_path, f"{cla_name}_shap_summary_stats.parquet")
    summary_stats.to_parquet(summary_path, engine='fastparquet')
    
    print(f"  [*] SHAP 数据已成功保存: {save_path}")


def export_model_response_data(model, X: pd.DataFrame, features: list, 
                               cla_name: str, output_path: str, y: pd.Series = None) -> None:
    """
    使用训练好的模型对全量数据进行预测，并导出拟合结果。
    """
    os.makedirs(output_path, exist_ok=True)

    if not isinstance(X, pd.DataFrame):
        raise TypeError("输入数据 X 必须是 pandas DataFrame")

    # 创建副本以保护原始数据并强制转换类型为 float32 (提升推理速度)
    X_processed = X.copy()
    for feature in features:
        X_processed[feature] = X_processed[feature].astype(np.float32)

    # 模型预测
    predicted_values = model.predict(X_processed)

    # 合并预测值与真实值
    response_data = X_processed.copy()
    response_data['predicted_WUE'] = predicted_values
    if y is not None:
        response_data['true_WUE'] = y.values
    
    # 保存结果
    output_file = os.path.join(output_path, f"{cla_name}_model_response_data.parquet")
    response_data.to_parquet(output_file, engine='fastparquet')

    print(f"  [*] 模型拟合响应数据已保存至: {output_file}")


# ==========================================
# 主流程：遍历各类气候区进行训练与评估
# ==========================================
if __name__ == "__main__":
    for i, file_path in enumerate(PARQUET_SET):
        current_cla = CLA_NAMES[i]
        seed = 20 + i  # 为每个数据集设置固定的随机种子，保证可复现性
        
        # ----------------------------------
        # 步骤 1: 数据加载与校验
        # ----------------------------------
        if not os.path.exists(file_path):
            print(f"\n[错误] 文件未找到 {file_path}，已跳过。")
            continue

        try:
            df = pd.read_parquet(file_path, engine='fastparquet')
        except Exception as e:
            print(f"\n[错误] 读取 {current_cla} 失败: {e}")
            continue
            
        print(f"\n=======================================================")
        print(f"开始处理数据集: {current_cla} (总样本数: {len(df)})")
        print(f"=======================================================")
        
        X = df[VARIABLES]
        y = df['WUE']
        
        # ----------------------------------
        # 步骤 2: 基于年份划分训练/测试集
        # ----------------------------------
        all_years = df["Year"].unique()
        n_years = len(all_years)
        n_train_years = N_TRAIN_YEARS_DEFAULT
        
        # 校验年份数量是否充足
        if n_years <= n_train_years:
            n_train_years = max(1, int(n_years * 0.8))  # 至少保留 1 年
            print(f"  [警告] 总年份({n_years})不足 15 年，动态调整训练集为 {n_train_years} 年。")

        # 随机抽取年份
        np.random.seed(seed)
        train_years = np.random.choice(all_years, size=n_train_years, replace=False)
        test_years = np.setdiff1d(all_years, train_years)

        # 提取对应数据索引
        train_idx = df.index[df["Year"].isin(train_years)]
        test_idx = df.index[df["Year"].isin(test_years)]

        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        # 转换为 numpy 数组并统一为 float32 (XGBoost GPU 加速推荐类型)
        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        y_train_np = y_train.values.astype(np.float32)
        y_test_np = y_test.values.astype(np.float32)
        
        print(f"  -> 训练集维度: {X_train.shape} | 测试集维度: {X_test.shape}")

        # ----------------------------------
        # 步骤 3: 贝叶斯优化寻找最优超参数
        # ----------------------------------
        kfold = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)

        def xgb_cv(n_estimators, max_depth, learning_rate, min_child_weight, 
                   reg_alpha, reg_lambda, gamma, subsample, max_delta_step, colsample_bytree):
            """XGBoost 交叉验证评估函数，服务于贝叶斯优化"""
            params = {
                'n_estimators': int(n_estimators),
                'max_depth': int(max_depth),
                'learning_rate': learning_rate,
                'min_child_weight': min_child_weight,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'gamma': gamma,
                'subsample': subsample,
                'max_delta_step': max_delta_step,
                'colsample_bytree': colsample_bytree,
                'random_state': seed,
                **GPU_CONFIG,
                'n_jobs': 1
            }
            model = xgb.XGBRegressor(**params)
            cv_scores = cross_val_score(
                model, X_train_np, y_train_np,
                cv=kfold,
                scoring='neg_root_mean_squared_error',
                n_jobs=1
            )
            return np.mean(cv_scores)

        # 贝叶斯优化的参数边界
        pbounds = {
            'n_estimators': (100, 1500),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'min_child_weight': (1, 10),
            'reg_alpha': (0, 10),
            'reg_lambda': (0, 10),
            'gamma': (0, 1),
            'max_delta_step': (0, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0)
        }
        
        optimizer = BayesianOptimization(f=xgb_cv, pbounds=pbounds, random_state=seed)
        
        print(f"\n  [1/4] 开始贝叶斯优化...")
        start_time = time.time()
        optimizer.maximize(init_points=10, n_iter=100)
        bayes_time = time.time() - start_time

        # 保存优化轨迹
        results_df = pd.DataFrame(optimizer.res)
        results_df.to_csv(os.path.join(OUT_DIR, f"{current_cla}_bayes_opt_results_seed{seed}.csv"), 
                          index=False, encoding='utf-8-sig')

        # 解析并格式化最优参数
        best_params = optimizer.max['params']
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        
        # 确保 JSON 序列化时的类型兼容性
        best_params = {
            k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else
               int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v
            for k, v in best_params.items()
        }
        
        final_params = {
            **best_params,
            **GPU_CONFIG,
            'n_jobs': 1,
            'random_state': seed,
        }
        
        # ----------------------------------
        # 步骤 4: 训练最终模型与评估
        # ----------------------------------
        print(f"\n  [2/4] 正在使用最优参数训练最终模型...")
        best_model = xgb.XGBRegressor(**final_params)

        start_time = time.time()
        best_model.fit(X_train_np, y_train_np)
        train_time = time.time() - start_time

        # 模型预测与指标计算
        y_train_pred = best_model.predict(X_train_np)
        y_test_pred = best_model.predict(X_test_np)

        metrics = {
            'train_r2': r2_score(y_train_np, y_train_pred),
            'test_r2': r2_score(y_test_np, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train_np, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_np, y_test_pred)),
            'train_mae': mean_absolute_error(y_train_np, y_train_pred),
            'test_mae': mean_absolute_error(y_test_np, y_test_pred)
        }
        
        print(f"  -> Train R2: {metrics['train_r2']:.4f} | Test R2: {metrics['test_r2']:.4f}")

        # 保存模型架构与参数
        model_pkl_path = os.path.join(OUT_DIR, f"{current_cla}_xgb_model_seed{seed}.pkl")
        model_json_path = os.path.join(OUT_DIR, f"{current_cla}_xgb_model_seed{seed}.json")
        joblib.dump(best_model, model_pkl_path)
        best_model.save_model(model_json_path)
        
        with open(model_pkl_path.replace('.pkl', '_params.json'), 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=4, ensure_ascii=False)

        # ----------------------------------
        # 步骤 5: SHAP 解释性分析
        # ----------------------------------
        real_samples = min(len(X), TARGET_SHAP_SAMPLES)
        print(f"\n  [3/4] 开始 SHAP 归因计算 (目标样本={TARGET_SHAP_SAMPLES}, 实际采用={real_samples})...")
        
        start_time = time.time()
        explainer = shap.TreeExplainer(best_model)
        
        if real_samples < TARGET_SHAP_SAMPLES:
            print(f"  [提示] 数据总行数小于设定采样数，将对全量数据进行计算。")
            shap_sample = X
        else:
            shap_sample = X.sample(n=real_samples, random_state=seed)

        shap_explanation = explainer(shap_sample)
        shap_values = shap_explanation.values
        base_values = explainer.expected_value
        shap_time = time.time() - start_time

        # ----------------------------------
        # 步骤 6: 结果保存与摘要生成
        # ----------------------------------
        print(f"\n  [4/4] 正在导出实验记录与数据文件...")
        
        # 将秒级耗时转换为 HH:MM:SS 格式的辅助函数
        def format_time(seconds):
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{int(h)}h {int(m)}m {s:.2f}s"

        # 写入运行摘要日志
        txt_path = os.path.join(OUT_DIR, f"{current_cla}_xgb_training_summary_seed{seed}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"==== {current_cla} XGBoost 实验总结 ====\n\n")
            f.write(f"随机种子: {seed}\n\n")
            
            f.write("---- 数据集概况 ----\n")
            f.write(f"总样本数: {len(X)} | 特征数: {X.shape[1]}\n")
            f.write(f"训练集: {len(X_train)} | 测试集: {len(X_test)}\n")
            f.write(f"训练集年份: {sorted(train_years)}\n")
            f.write(f"测试集年份: {sorted(test_years)}\n\n")
            
            f.write("---- 耗时统计 ----\n")
            f.write(f"贝叶斯优化耗时: {format_time(bayes_time)}\n")
            f.write(f"模型训练耗时: {format_time(train_time)}\n")
            f.write(f"SHAP计算耗时: {format_time(shap_time)}\n\n")
            
            f.write("---- 模型性能 ----\n")
            f.write(f"Train R²: {metrics['train_r2']:.4f}, RMSE: {metrics['train_rmse']:.4f}, MAE: {metrics['train_mae']:.4f}\n")
            f.write(f"Test R²:  {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}, MAE: {metrics['test_mae']:.4f}\n\n")
            
            f.write("---- 最优超参数 ----\n")
            f.write(json.dumps(best_params, indent=4, ensure_ascii=False) + "\n\n")
            
            f.write("---- SHAP 分析配置 ----\n")
            f.write(f"设定目标采样点: {TARGET_SHAP_SAMPLES} | 实际参与计算点: {real_samples}\n")

        # 导出各类数据 parquet 文件
        export_shap_data(shap_values, shap_sample, y.loc[shap_sample.index], current_cla, OUT_DIR, base_values=base_values)
        export_model_response_data(best_model, X, VARIABLES, current_cla, OUT_DIR, y)
        
        print(f"{current_cla} 区域的所有任务已圆满完成！\n")