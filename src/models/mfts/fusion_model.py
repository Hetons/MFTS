import torch
import torch.nn as nn
from mfts_refine_model import (
    MtfsRefineModel,
    ShardedGraphDataset as PayloadShardedGraphDataset,
)
from mfts_early_model import (
    MftsEarlyModel,
    ShardedGraphDataset as TlsShardedGraphDataset,
)
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnifiedData:
    def __init__(self, payload_data, tls_data):
        self.payload_data = payload_data
        self.tls_data = tls_data

    def to(self, device):
        self.payload_data = self.payload_data.to(device)
        tls_data_x, y = self.tls_data
        # tls_data_x 增加 batch 维度
        self.tls_data_x = tls_data_x.unsqueeze(0).to(device)
        self.payload_x = self.payload_data.x.to(device)
        self.payload_edge_index = self.payload_data.edge_index.to(device)
        self.payload_edge_attr = self.payload_data.edge_attr.to(device)
        self.payload_batch = self.payload_data.batch
        if self.payload_batch is not None:
            self.payload_batch = self.payload_batch.to(device)
        self.y = y.to(device)
        return self


class FusionModelDataset(Dataset):
    def __init__(self, root_dir: str, num_classes: int):
        self.payload_dataset = PayloadShardedGraphDataset(
            root=root_dir, num_classes=num_classes
        )
        self.tls_dataset = TlsShardedGraphDataset(root=root_dir)

    def __len__(self):
        return len(self.payload_dataset)

    def __getitem__(self, idx):
        payload_data = self.payload_dataset[idx]
        tls_data = self.tls_dataset[idx]
        return UnifiedData(payload_data, tls_data).to(device)


class FusionModel(nn.Module):
    def __init__(
        self,
        quick_ratio: float = 0.99,
        tls_model_path: str = "",
        payload_model_path: str = "",
    ):
        super().__init__()
        self.quick_ratio = quick_ratio
        self.tls_model: MftsEarlyModel
        self.payload_model: MtfsRefineModel
        self.tls_model, self.payload_model = self.load_model(
            payload_model_path, tls_model_path
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, unified_data: UnifiedData):
        """
        :param self: fusion model 实例
        :param t: tls header 输入
        :param x: payload node features 输入
        :param edge_index: edge index 输入
        :param edge_attr: edge attributes 输入
        :param batch: batch 输入
        :return: 最终的 logits 输出
        """
        # tls classifier
        tls_pred = self.tls_model(unified_data.tls_data_x)
        tls_probs = self.softmax(tls_pred)
        max_prob, pred = torch.max(tls_probs, dim=1)
        if max_prob >= self.quick_ratio:
            return tls_pred

        # payload classifier
        payload_pred = self.payload_model(
            unified_data.payload_x,
            unified_data.payload_edge_index,
            unified_data.payload_edge_attr,
            unified_data.payload_batch,
        )
        final_probs = 0.6 * self.softmax(payload_pred) + 0.4 * tls_probs
        return torch.log(final_probs + 1e-8)  # 转回 logits

    def load_model(
        self, payload_model_path: str, tls_model_path: str
    ) -> tuple[MftsEarlyModel, MtfsRefineModel]:
        tls_model_check_point = torch.load(
            tls_model_path, map_location=device, weights_only=False
        )
        tls_model = MftsEarlyModel(
            class_num=tls_model_check_point["config"]["num_classes"],
            d_model=tls_model_check_point["config"]["input_dim"],
            hidden_dim=tls_model_check_point["config"]["hidden_dim"],
            dropout_param=tls_model_check_point["config"]["dropout_param"],
        ).to(device)
        tls_model.load_state_dict(tls_model_check_point["model_state_dict"])

        payload_model_check_point = torch.load(
            payload_model_path, map_location=device, weights_only=False
        )
        payload_model = MtfsRefineModel(
            in_dim=payload_model_check_point["config"]["in_dim"],
            hidden_dim=payload_model_check_point["config"]["hidden_dim"],
            num_classes=payload_model_check_point["config"]["num_classes"],
            heads=payload_model_check_point["config"]["heads"],
            dropout_param=payload_model_check_point["config"]["dropout_param"],
        ).to(device)
        payload_model.load_state_dict(payload_model_check_point["model_state_dict"])

        return tls_model, payload_model


class FusionModelEvaluator:

    # ACC / F1 / Precision / Recall / Inference Time / Matrix / Parameters / FLOPs
    def evaluate(self, model: FusionModel, data: UnifiedData):
        import torch.profiler as prof

        x = data.to(device)

        def run_infer(model, x, steps=50):
            model.eval()
            with torch.inference_mode():
                for _ in range(steps):
                    _ = model(x)

        # 建议：先做纯 warmup（不进 profiler）
        run_infer(model, x, steps=20)
        torch.cuda.synchronize()

        schedule = prof.schedule(wait=2, warmup=2, active=10, repeat=1)

        with prof.profile(
            activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.CUDA],
            schedule=schedule,
            record_shapes=True,  # 想看 shape 才开；会更重
            profile_memory=True,  # 想看显存/内存才开；也更重
            with_stack=False,  # 推理默认关
            on_trace_ready=prof.tensorboard_trace_handler("./tb_logs"),
        ) as p:
            model.eval()
            with torch.inference_mode():
                for _ in range(2 + 2 + 10):  # wait+warmup+active
                    _ = model(x)
                    p.step()

        torch.cuda.synchronize()

        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=30))


"""这里融合了 Payload 模型和 TLS Header 模型，两阶段分类的具体实现"""
if __name__ == "__main__":
    test_dataset_ratio = 0.2
    dataset = FusionModelDataset(
        root_dir="/home/tyf/Project/Tantic/raw_feature/stgc_sp_all_class_tls_2",
        num_classes=17,
    )
    n = len(dataset)
    n_val = int(test_dataset_ratio * n)
    n_train = n - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    model = FusionModel(
        quick_ratio=0.99,
        tls_model_path="/home/tyf/Project/Tantic/checkpoints/fast_tls_cnn.pth",
        payload_model_path="/home/tyf/Project/Tantic/checkpoints/payload_gnn_model.pth",
    )
    all_y_pred = []
    all_y_true = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            if isinstance(data, UnifiedData):
                y_pred = model(data)
                all_y_pred.append(y_pred.argmax(dim=1).cpu())
                all_y_true.append(data.y.unsqueeze(0).cpu())
    all_y_pred = torch.cat(all_y_pred, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)

    from sklearn.metrics import classification_report

    # Performance evaluation
    data = dataset[0]
    evaluator = FusionModelEvaluator()
    evaluator.evaluate(model, data)

    print(classification_report(all_y_true.numpy(), all_y_pred.numpy(), digits=4))
