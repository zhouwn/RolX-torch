import pandas as pd
import numpy as np

# --- 配置 ---
embedding_file = r'D:\experiment\network dismantling\RolX-torch\output\embeddings\Douban_embedding.csv'  # 输入的embedding文件名
hard_assignment_file = r'D:\experiment\network dismantling\RolX-torch\output\embeddings\Douban_node_roles.csv' # 输出的精确角色文件名
one_hot_output_file = r'D:\experiment\network dismantling\RolX-torch\output\embeddings\Douban_node_roles_one_hot.csv' # 输出的one-hot编码文件名

# --- 读取数据 ---
try:
    df_embedding = pd.read_csv(embedding_file)
    # 将列名统一为 role_0, role_1, ...
    df_embedding.columns = [f"role_{i}" for i in range(df_embedding.shape[1])]
except FileNotFoundError:
    print(f"错误：找不到文件 '{embedding_file}'。请确保文件存在于正确的位置。")
    exit()

# --- 计算精确角色 ---
# idxmax(axis=1) 会返回每一行最大值的列名
precise_roles = df_embedding.idxmax(axis=1)
# 提取角色编号，例如从 "role_2" 中提取 2
role_ids = precise_roles.str.split('_').str[1].astype(int)

# 创建一个新的DataFrame来保存结果
df_roles = pd.DataFrame({
    'node_id': df_embedding.index,
    'assigned_role': role_ids
})

# --- 保存精确角色分配结果 ---
df_roles.to_csv(hard_assignment_file, index=False)
print(f"精确的角色分配已保存到 '{hard_assignment_file}'")
print(df_roles.head())

# --- (可选) 生成并保存 one-hot 编码结果 ---
# np.argmax(df_embedding.values, axis=1) 找到每行最大值的索引
max_indices = np.argmax(df_embedding.values, axis=1)

# 创建一个全零矩阵
one_hot_matrix = np.zeros_like(df_embedding.values, dtype=int)

# 将最大值索引位置设为1
one_hot_matrix[np.arange(len(df_embedding)), max_indices] = 1

df_one_hot = pd.DataFrame(one_hot_matrix, columns=df_embedding.columns)

# 保存 one-hot 编码结果
df_one_hot.to_csv(one_hot_output_file, index=False)
print(f"\nOne-hot 编码的角色分配已保存到 '{one_hot_output_file}'")
print(df_one_hot.head())