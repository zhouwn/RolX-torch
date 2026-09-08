# rolx.py
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from layers import Factorization
from refex import RecursiveExtractor
from print_and_read import log_setup, tab_printer, epoch_printer, log_updater, data_saver


class ROLX:
    """
    ROLX class for PyTorch implementation.
    """

    def __init__(self, args):
        """
        Model setup: feature extraction, model, optimizer, and loss function initialization.
        """
        self.args = args
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Feature Extraction (framework-agnostic)
        self.recurser = RecursiveExtractor(args)
        self.dataset = self.recurser.new_features.values.astype(np.float32)

        self.user_size = self.dataset.shape[0]
        self.feature_size = self.dataset.shape[1]
        self.nodes = list(range(self.user_size))

        # 2. Build PyTorch Model
        self.model = Factorization(self.args, self.user_size, self.feature_size).to(self.device)

        # 3. Setup Loss and Optimizer
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.initial_learning_rate)

        # 4. Setup Learning Rate Scheduler
        num_batches = len(self.nodes) // self.args.batch_size
        self.true_step_size = self.args.epochs * num_batches
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=self.true_step_size,
            power=self.args.annealing_factor,
            verbose=False
        )

    def prepare_batch(self, nodes):
        """
        Prepares a batch of data and moves it to the target device.
        """
        left_nodes = torch.LongTensor(nodes).to(self.device)
        right_nodes = torch.LongTensor(range(self.feature_size)).to(self.device)

        # Targets are the corresponding rows from the feature matrix
        targets = torch.from_numpy(self.dataset[nodes, :]).to(self.device)

        return left_nodes, right_nodes, targets

    def train(self):
        """
        Main training loop for the ROLX model.
        """
        self.log = log_setup(self.args)
        print("Model Initialized and Training Started.")

        for epoch in range(self.args.epochs):
            random.shuffle(self.nodes)
            epoch_printer(epoch)

            start_time = time.time()
            total_loss = 0

            num_batches = len(self.nodes) // self.args.batch_size
            if num_batches == 0:
                print("Warning: Batch size is larger than the number of nodes. No batches to process.")
                continue

            batch_iterator = tqdm(range(num_batches))
            for i in batch_iterator:
                # 1. Prepare Batch
                batch_nodes = self.nodes[i * self.args.batch_size: (i + 1) * self.args.batch_size]
                left_nodes, right_nodes, targets = self.prepare_batch(batch_nodes)

                # 2. Forward pass
                predictions = self.model(left_nodes, right_nodes)

                # 3. Calculate loss
                loss = self.loss_fn(predictions, targets)

                # 4. Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 5. Update learning rate
                self.scheduler.step()

                total_loss += loss.item()
                batch_iterator.set_description(f"Epoch {epoch + 1}/{self.args.epochs} | Loss: {loss.item():.4f}")

            optimization_time = time.time() - start_time
            average_loss = total_loss / num_batches

            # Logging
            self.log = log_updater(self.log, epoch, average_loss, optimization_time)
            tab_printer(self.log)

        # Save the final node embeddings
        print("Training finished. Saving embeddings.")
        # Set model to evaluation mode
        self.model.eval()
        # Get embeddings, move to CPU, detach from graph, and convert to numpy
        final_embeddings = self.model.embedding_node.weight.data.cpu().detach().numpy()
        data_saver(final_embeddings, self.args.embedding_output)

        # --- 新增代码开始 ---
        print("正在保存角色-特征嵌入...")
        # 获取特征嵌入 (F矩阵)
        final_feature_embeddings = self.model.embedding_feature.weight.data.cpu().detach().numpy()

        # 将其保存为DataFrame，以便包含特征名称
        # 从 recurser 获取原始特征的列名
        try:
            # 获取特征嵌入矩阵，其形状为 (特征数, 角色数)，例如 (14, 48)
            final_feature_embeddings = self.model.embedding_feature.weight.data.cpu().detach().numpy()

            # 获取特征名称列表，其长度应与特征数相同，例如 14
            feature_columns = self.recurser.new_features.columns

            # 直接使用原始矩阵创建DataFrame，不进行转置(.T)
            # 这样数据的行数 (14) 就和索引的长度 (14) 匹配了
            df_feature_embeddings = pd.DataFrame(
                final_feature_embeddings,
                index=feature_columns,
                columns=[f'role_{i}' for i in range(self.args.dimensions)]
            )

            df_feature_embeddings.to_csv(r'D:\experiment\network dismantling\RolX-torch\output\embeddings\Douban_role_features.csv', index_label='feature_name')
            print("角色特征已保存到 'Douban_role_features.csv'")

        except Exception as e:
            print(f"创建或保存角色特征文件时出错: {e}")
            print("将尝试以无头CSV格式保存原始矩阵。")
            np.savetxt(r'D:\experiment\network dismantling\RolX-torch\output\embeddings\Douban_role_features.csv', final_feature_embeddings, delimiter=",")
        # --- 替换结束 ---
