import torch
import torch.nn as nn
from tantic_payload_model import (
    GATGraphClassifier,
    ShardedGraphDataset as PayloadShardedGraphDataset,
)
from tantic_tls_header_model import (
    FastTlsCNN,
    ShardedGraphDataset as TlsShardedGraphDataset,
)
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return payload_data, tls_data


class FusionModel(nn.Module):
    def __init__(
        self,
        quick_ratio: float = 0.99,
        tls_model_path: str = "",
        payload_model_path: str = "",
    ):
        super().__init__()
        self.quick_ratio = quick_ratio
        self.tls_model: FastTlsCNN
        self.payload_model: GATGraphClassifier
        self.tls_model, self.payload_model = self.load_model(
            payload_model_path, tls_model_path
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, t, x, edge_index, edge_attr, batch):
        # tls classifier
        tls_pred = self.tls_model(t)
        tls_probs = self.softmax(tls_pred)
        max_prob, pred = torch.max(tls_probs, dim=1)
        if max_prob >= self.quick_ratio:
            return tls_pred

        # payload classifier
        payload_pred = self.payload_model(x, edge_index, edge_attr, batch)
        final_probs = 0.6 * self.softmax(payload_pred) + 0.4 * tls_probs
        return torch.log(final_probs + 1e-8)  # 转回 logits

    def load_model(
        self, payload_model_path: str, tls_model_path: str
    ) -> tuple[FastTlsCNN, GATGraphClassifier]:
        tls_model_check_point = torch.load(
            tls_model_path, map_location=device, weights_only=False
        )
        tls_model = FastTlsCNN(
            class_num=tls_model_check_point["config"]["num_classes"],
            d_model=tls_model_check_point["config"]["input_dim"],
            hidden_dim=tls_model_check_point["config"]["hidden_dim"],
            dropout_param=tls_model_check_point["config"]["dropout_param"],
        ).to(device)
        tls_model.load_state_dict(tls_model_check_point["model_state_dict"])

        payload_model_check_point = torch.load(
            payload_model_path, map_location=device, weights_only=False
        )
        payload_model = GATGraphClassifier(
            in_dim=payload_model_check_point["config"]["in_dim"],
            hidden_dim=payload_model_check_point["config"]["hidden_dim"],
            num_classes=payload_model_check_point["config"]["num_classes"],
            heads=payload_model_check_point["config"]["heads"],
            dropout_param=payload_model_check_point["config"]["dropout_param"],
        ).to(device)
        payload_model.load_state_dict(payload_model_check_point["model_state_dict"])

        return tls_model, payload_model


class FusionModelEvaluator:
    def __init__(self):
        pass

    # ACC / F1 / Precision / Recall / Inference Time / Matrix / Parameters / FLOPs
    def evaluate(self, model: FusionModel, dataset: FusionModelDataset):
        pass

    pass


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
            if isinstance(data, tuple):
                payload_data, tls_data = data
                tls_data_x, y = tls_data
                # tls_data_x 增加 batch 维度
                tls_data_x = tls_data_x.unsqueeze(0).to(device)
                y = y.to(device)
                payload_data = payload_data.to(device)
                y_pred = model(
                    tls_data_x,
                    payload_data.x,
                    payload_data.edge_index,
                    payload_data.edge_attr,
                    payload_data.batch,
                )
                all_y_pred.append(y_pred.argmax(dim=1).cpu())
                all_y_true.append(y.unsqueeze(0).cpu())
    all_y_pred = torch.cat(all_y_pred, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)

    from sklearn.metrics import classification_report

    print(classification_report(all_y_true.numpy(), all_y_pred.numpy(), digits=4))
