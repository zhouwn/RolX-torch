# rolx.py 关键替换片段

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
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1) 特征抽取
        self.recurser = RecursiveExtractor(args)
        self.dataset = self.recurser.new_features.values.astype(np.float32)
        self.user_size = self.dataset.shape[0]
        self.feature_size = self.dataset.shape[1]
        self.nodes = list(range(self.user_size))

        if self.feature_size == 0:
            print("[警告] 无特征，注入全零占位列。")
            self.dataset = np.zeros((self.user_size, 1), dtype=np.float32)
            self.feature_size = 1

        # 预分配最终节点/特征嵌入的容器（在 CPU 上）
        self.total_dims = self.args.dimensions
        self.stage_dim = max(1, int(getattr(self.args, "stage_dim", min(512, self.total_dims))))
        self.num_stages = (self.total_dims + self.stage_dim - 1) // self.stage_dim

        self.final_node_embed_cpu = np.zeros((self.user_size, self.total_dims), dtype=np.float32)
        self.final_feat_embed_cpu = np.zeros((self.feature_size, self.total_dims), dtype=np.float32)

    def _build_model_and_opt(self, stage_dim):
        # 按阶段维度构建模型
        from layers import Factorization
        class ArgsView:
            # 只把本阶段需要的维度注入
            def __init__(self, base_args, stage_dim):
                self.__dict__.update(vars(base_args))
                self.dimensions = stage_dim
        stage_args = ArgsView(self.args, stage_dim)

        model = Factorization(stage_args, self.user_size, self.feature_size).to(self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        # 用 Adam 也可以，但为省状态内存，建议 Adagrad 或者 Adam(注意显存)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=self.args.initial_learning_rate)
        return model, loss_fn, optimizer

    def prepare_batch(self, nodes):
        left_nodes = torch.LongTensor(nodes).to(self.device)
        right_nodes = torch.LongTensor(range(self.feature_size)).to(self.device)
        targets = torch.from_numpy(self.dataset[nodes, :]).to(self.device)
        return left_nodes, right_nodes, targets

    def train(self):
        self.log = log_setup(self.args)
        print("Model Initialized and Training Started.")

        # ====== 多阶段循环 ======
        filled = 0
        for s in range(self.num_stages):
            cur_dim = min(self.stage_dim, self.total_dims - filled)
            print(f"\n[Stage {s+1}/{self.num_stages}] 训练 {cur_dim} 维（总 {self.total_dims} 维中的 {filled}~{filled+cur_dim-1}）")

            # 1) 构建模型与优化器（仅 cur_dim 维）
            model, loss_fn, optimizer = self._build_model_and_opt(cur_dim)

            # 2) 为本阶段创建 scheduler（拿到真实 num_batches 之后）
            scheduler = None

            for epoch in range(self.args.epochs):
                random.shuffle(self.nodes)
                epoch_printer(epoch)

                start_time = time.time()
                total_loss = 0.0

                bs = max(1, self.args.batch_size)
                num_batches = max(1, len(self.nodes) // bs)

                if scheduler is None:
                    total_iters = max(1, self.args.epochs * num_batches)
                    scheduler = torch.optim.lr_scheduler.PolynomialLR(
                        optimizer, total_iters=total_iters,
                        power=max(1e-6, float(self.args.annealing_factor)),
                        verbose=False
                    )

                batch_iter = tqdm(range(num_batches))
                for i in batch_iter:
                    start = i * bs
                    end = len(self.nodes) if i == num_batches - 1 else (i + 1) * bs
                    batch_nodes = self.nodes[start:end]
                    left_nodes, right_nodes, targets = self.prepare_batch(batch_nodes)

                    logits = model(left_nodes, right_nodes)
                    loss = loss_fn(logits, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += float(loss.item())
                    batch_iter.set_description(f"Stage {s+1} | Epoch {epoch+1}/{self.args.epochs} | Loss: {loss.item():.4f}")

                optimization_time = time.time() - start_time
                average_loss = total_loss / max(1, num_batches)
                self.log = log_updater(self.log, epoch, average_loss, optimization_time)
                tab_printer(self.log)

            # 3) 本阶段结束，拉取权重到 CPU，放入最终大矩阵
            model.eval()
            with torch.no_grad():
                node_embed = model.embedding_node.weight.detach().cpu().numpy()        # (N, cur_dim)
                feat_embed = model.embedding_feature.weight.detach().cpu().numpy()     # (F, cur_dim)
                self.final_node_embed_cpu[:, filled:filled+cur_dim] = node_embed
                self.final_feat_embed_cpu[:, filled:filled+cur_dim] = feat_embed

            # 4) 释放显存
            del model, node_embed, feat_embed
            torch.cuda.empty_cache()

            filled += cur_dim

        # ====== 全部阶段完成：保存 ======
        print("Training finished. Saving embeddings.")
        from print_and_read import data_saver
        data_saver(self.final_node_embed_cpu, self.args.embedding_output)

        print("正在保存角色-特征嵌入...")
        feature_columns = getattr(self.recurser.new_features, "columns",
                                  [f"f_{i}" for i in range(self.feature_size)])
        df_feature_embeddings = pd.DataFrame(
            self.final_feat_embed_cpu,
            index=feature_columns,
            columns=[f'role_{i}' for i in range(self.total_dims)]
        )
        role_feat_out = self.args.embedding_output.replace("_embedding", "_role_features")
        df_feature_embeddings.to_csv(role_feat_out, index_label='feature_name')
        print(f"角色特征已保存到 '{role_feat_out}'")
