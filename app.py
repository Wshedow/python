from flask import Flask, request, jsonify, send_file, session, redirect
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import chardet
import matplotlib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import uuid
import functools
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import json
import getpass

# 自定义JSON编码器，处理NumPy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return float('inf') if obj > 0 else float('-inf')
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 处理可能包含Infinity或NaN的值，确保JSON可序列化
def safe_json_value(val):
    if isinstance(val, (np.integer, np.floating)):
        if np.isnan(val):
            return None
        elif np.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        return float(val)
    elif isinstance(val, np.ndarray):
        return [safe_json_value(x) for x in val]
    return val

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 增加到64MB
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/pythonsy'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key_here'  # 用于会话加密
app.json_encoder = NumpyEncoder  # 使用自定义JSON编码器
db = SQLAlchemy(app)

# 添加模板目录配置
app.template_folder = 'templates'

# 创建uploads目录，用于存储上传的文件
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
@login_required
def upload_file(user_id, username, user_type):
    try:
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {save_path}")
        
        # 确保上传目录存在
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # 增加错误处理
        try:
            file.save(save_path)
            print("File saved successfully")
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({'error': f'文件保存失败: {str(e)}'}), 500
        
        # 检查文件大小
        file_size = os.path.getsize(save_path)
        if file_size > 50 * 1024 * 1024:  # 50MB
            print(f"Large file detected: {file_size / (1024 * 1024):.2f} MB")
        
        # 根据文件扩展名选择正确的读取方法
        try:
            if filename.lower().endswith('.csv'):
                # 对于CSV文件，尝试检测编码
                with open(save_path, 'rb') as f:
                    raw = f.read(10000)  # 只读取前10KB用于检测编码
                    result = chardet.detect(raw)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
                print(f"Detected encoding: {encoding}")
                try:
                    # 使用chunksize读取大文件
                    df_chunks = pd.read_csv(save_path, encoding=encoding, on_bad_lines='skip', chunksize=1000)
                    df = next(df_chunks)  # 只获取第一个chunk用于预览
                    total_rows = sum(1 for _ in pd.read_csv(save_path, encoding=encoding, on_bad_lines='skip', iterator=True, chunksize=1000))
                except Exception as e:
                    print(f"Error reading CSV: {e}")
                    return jsonify({'error': f'CSV文件读取失败：{e}，请用Excel另存为UTF-8编码后重试。'}), 400
            elif filename.lower().endswith(('.xls', '.xlsx')):
                # 对于Excel文件，使用openpyxl引擎
                try:
                    # 对于大文件，可能需要设置IO相关参数
                    df = pd.read_excel(save_path, engine='openpyxl', nrows=1000)  # 仅读取前1000行用于预览
                    total_rows = pd.read_excel(save_path, engine='openpyxl', usecols=[0], header=None).shape[0]  # 获取总行数
                    print("Successfully read Excel file")
                except Exception as e:
                    print(f"Error reading Excel: {e}")
                    return jsonify({'error': f'Excel文件读取失败：{e}'}), 400
            else:
                return jsonify({'error': '不支持的文件格式，请上传CSV或Excel文件'}), 400
            
            print(f"Data shape: {df.shape}")
            print(f"Total rows in file: {total_rows}")
            
            # 记录文件上传操作
            log_user_action(username, user_id, "上传文件", 
                           filename=filename, 
                           details=f"文件大小: {os.path.getsize(save_path)} 字节, 数据行数: {total_rows}, 列数: {df.shape[1]}")
            
            # 准备数据以安全序列化为JSON
            # 替换NaN为None (在JSON中会转为null)
            df = df.astype(object).where(pd.notnull(df), None)
            
            # 获取列名
            columns = df.columns.tolist()
            
            # 最多返回1000行预览，确保UI能够处理
            max_preview_rows = min(1000, len(df))
            
            # 使用.to_dict('records')获取行字典列表，更好地处理复杂类型
            rows_data = df.head(max_preview_rows).to_dict('records')
            
            # 将行字典列表转换为行值列表，确保所有值可序列化
            rows = []
            for row_dict in rows_data:
                row = []
                for col in columns:
                    val = row_dict.get(col)
                    # 处理非原始类型
                    if not (val is None or isinstance(val, (str, int, float, bool))):
                        val = str(val)
                    row.append(val)
                rows.append(row)
            
            # 返回预览数据，使用jsonify自动处理序列化
            preview = {
                'columns': columns,
                'rows': rows,
                'total_rows': total_rows,
                'total_columns': len(columns),
                'filename': filename,  # 添加文件名字段到响应中
                'preview_rows': max_preview_rows  # 添加预览的行数信息
            }
            
            # 确保正确设置content-type和字符编码
            response = jsonify(preview)
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"File processing error: {e}")
            return jsonify({'error': f'文件处理出错: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'上传文件出错: {str(e)}'}), 500
def plot_cluster_analysis(X_scaled, clusters, kmeans, wcss, silhouette_scores, n_clusters, 
                         numeric_cols, df, filename, feature_importance, 
                         optimal_clusters_elbow, optimal_clusters_silhouette):
    """将聚类分析图表生成逻辑提取到单独的函数"""
    try:
        # 确保使用全局变量
        global UPLOAD_FOLDER
        
        # 创建子图布局，为结果创建更多视图
        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 1. 聚类散点图
        ax1 = axes[0, 0]
        if X_scaled.shape[1] >= 2:
            # 获取特征重要性最高的两个特征索引
            top_features = list(feature_importance.keys())[:2] if len(feature_importance) >= 2 else numeric_cols[:2]
            top_feature_indices = [numeric_cols.index(f) for f in top_features]
            
            # 绘制散点图
            scatter = ax1.scatter(X_scaled[:, top_feature_indices[0]], 
                               X_scaled[:, top_feature_indices[1]], 
                               c=clusters, cmap='viridis', alpha=0.7)
            
            # 绘制聚类中心
            centers = kmeans.cluster_centers_
            ax1.scatter(centers[:, top_feature_indices[0]], 
                     centers[:, top_feature_indices[1]],
                     s=200, c='red', marker='X', label='聚类中心')
            
            ax1.set_title('主要特征二维聚类图')
            ax1.set_xlabel(top_features[0])
            ax1.set_ylabel(top_features[1])
            ax1.legend()
            
            # 添加图例
            legend1 = ax1.legend(*scatter.legend_elements(), title="聚类组")
            ax1.add_artist(legend1)
        else:
            ax1.text(0.5, 0.5, '特征数量不足，无法生成二维散点图', ha='center', va='center')
        
        # 2. 肘部法则图，帮助确定最佳聚类数
        ax2 = axes[0, 1]
        x_range = range(2, 2 + len(wcss))
        ax2.plot(x_range, wcss, marker='o', linestyle='-', color='blue', label='WCSS')
        
        # 标记最优聚类数（肘部法则）
        if optimal_clusters_elbow:
            ax2.axvline(x=optimal_clusters_elbow, color='r', linestyle='--', 
                      label=f'最优聚类数(肘部法则): {optimal_clusters_elbow}')
        
        # 标记当前选择的聚类数
        ax2.axvline(x=n_clusters, color='g', linestyle=':', 
                  label=f'当前选择: {n_clusters}')
        
        ax2.set_title('肘部法则图 - 确定最佳聚类数')
        ax2.set_xlabel('聚类数量')
        ax2.set_ylabel('WCSS (组内平方和)')
        ax2.legend()
        
        # 3. 轮廓系数图，另一种评估最佳聚类数的方法
        ax3 = axes[1, 0]
        x_range = range(2, 2 + len(silhouette_scores))
        ax3.plot(x_range, silhouette_scores, marker='o', linestyle='-', color='orange')
        
        # 标记最优聚类数（轮廓系数）
        if optimal_clusters_silhouette:
            ax3.axvline(x=optimal_clusters_silhouette, color='r', linestyle='--', 
                      label=f'最优聚类数(轮廓系数): {optimal_clusters_silhouette}')
        
        # 标记当前选择的聚类数
        ax3.axvline(x=n_clusters, color='g', linestyle=':', 
                  label=f'当前选择: {n_clusters}')
        
        ax3.set_title('轮廓系数图 - 另一种评估最佳聚类数的方法')
        ax3.set_xlabel('聚类数量')
        ax3.set_ylabel('轮廓系数 (越高越好)')
        ax3.legend()
        
        # 4. 聚类分布图
        ax4 = axes[1, 1]
        cluster_counts = np.bincount(clusters)
        bars = ax4.bar(range(len(cluster_counts)), cluster_counts, color='skyblue')
        ax4.set_title('各聚类数据量分布')
        ax4.set_xlabel('聚类组')
        ax4.set_ylabel('数据点数量')
        ax4.set_xticks(range(len(cluster_counts)))
        
        # 在柱子上显示具体数值
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}',
                    ha='center', va='bottom')
        
        # 5. 特征重要性图
        ax5 = axes[2, 0]
        
        # 提取特征重要性得分并处理None值
        features = list(feature_importance.keys())
        importance_scores = []
        for f in features:
            score = feature_importance[f]['importance']
            if score is None or isinstance(score, str):
                importance_scores.append(0.0)  # 将None或字符串值替换为0
            else:
                importance_scores.append(float(score))
        
        # 按重要性排序，并显示前10个特征
        sorted_indices = np.argsort(importance_scores)[::-1]
        top_n = min(10, len(features))
        
        top_features = [features[i] for i in sorted_indices[:top_n]]
        top_scores = [importance_scores[i] for i in sorted_indices[:top_n]]
        
        # 创建水平条形图
        bars = ax5.barh(range(len(top_features)), top_scores, color='lightgreen')
        ax5.set_yticks(range(len(top_features)))
        ax5.set_yticklabels(top_features)
        ax5.set_title('特征重要性')
        ax5.set_xlabel('重要性分数 (%)')
        
        # 在条形上显示数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax5.text(width + 1, i, f'{width:.1f}%', va='center')
        
        # 6. 各聚类特征均值热力图
        ax6 = axes[2, 1]
        if len(numeric_cols) > 1:
            # 计算每个聚类的特征均值
            cluster_means = df.groupby('Cluster')[numeric_cols].mean()
            
            # 创建热力图
            sns.heatmap(cluster_means, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax6)
            ax6.set_title('聚类特征均值热力图')
            ax6.set_ylabel('聚类组')
        else:
            ax6.text(0.5, 0.5, '特征数量不足，无法生成热力图', ha='center', va='center')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像到文件
        img_path = os.path.join(UPLOAD_FOLDER, f'cluster_{filename}.png')
        plt.savefig(img_path, format='png', dpi=100)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error generating cluster plots: {e}")
        import traceback
        traceback.print_exc()
        return False
