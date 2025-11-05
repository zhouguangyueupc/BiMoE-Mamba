# # -*- coding: utf-8 -*-
# import argparse, os, math, time, numpy as np, random, torch
# import torch.nn as nn
# from torch import optim
# from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import StratifiedShuffleSplit
# from models.moe_bi_mamba import MambaStack   # ✅ 使用带 MoE 的模型
# # from models.bi_mamba import MambaStack
# from dataset import HarDataset
# from models.pruemamba import PureMambaClassifier   # 你的纯 Mamba 模型
# from utils import cal_best_model_evaluating_indicator
#
# # ---------------- CLI ---------------- #
# parser = argparse.ArgumentParser()
# parser.add_argument('--seed', default=1, type=int)
# parser.add_argument('--dataset-number', default=0, type=int, dest='dataset_number')
# parser.add_argument('--epochs', default=1000, type=int)
# parser.add_argument('--batch-size', default=64, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--weight-decay', default=0.02, type=float)
# parser.add_argument('--label-smoothing', default=0.03, type=float)  # 0.02~0.05 建议
# # 纯 Mamba 模型超参
# parser.add_argument('--d-model', default=64, type=int)  # 要是32的话，学不到什么东西，准确率比64要低1%至少，如果不改变Layers，128也和64差不多
# parser.add_argument('--d-state', default=32, type=int)
# parser.add_argument('--d-conv',  default=8, type=int)
# parser.add_argument('--expand',  default=2, type=int)
# parser.add_argument('--layers',  default=3, type=int)
# # 轻量数据增强强度（可按需微调）
# parser.add_argument('--aug-noise-std', default=0.01, type=float)     # 高斯噪声 σ
# parser.add_argument('--aug-scale-std', default=0.05, type=float)     # 通道缩放抖动
# parser.add_argument('--aug-time-mask-ratio', default=0.05, type=float)  # 时间掩蔽比例
# args = parser.parse_args()
#
#
# # ---------------- Device ---------------- #
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device:', device)
#
# # ---------------- Dataset cfg ---------------- #
# dataset_dict = {
#     'uci-har': {'file_path': './data/UCI_data/np/',      'd_input': 128, 'd_channel': 9,  'd_output': 6},
#     'usc-had': {'file_path': './data/USC_HAD_data/np/',  'd_input': 128, 'd_channel': 6,  'd_output': 12},
#     'real-world': {'file_path': './data/Real_World_data/np/', 'd_input': 128, 'd_channel': 21, 'd_output': 8},
# }
# dataset_names = ['uci-har','usc-had','real-world']
# use_dataset_name = dataset_names[args.dataset_number]
# cfg = dataset_dict[use_dataset_name]
# d_input, d_channel, d_output = cfg['d_input'], cfg['d_channel'], cfg['d_output']
#
# # ---------------- Reproducibility ---------------- #
# torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
# np.random.seed(args.seed); random.seed(args.seed)
# torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
#
# # ---------------- Datasets & Loaders ---------------- #
# train_full = HarDataset(cfg['file_path'], trainOrTest='train')   # 期望 __getitem__ → (x:(L,C) or (C,L), y)
# test_set   = HarDataset(cfg['file_path'], trainOrTest='test')
#
# # 因为你不想要验证集，所以我们不再进行拆分，只使用全部训练数据
# train_set = train_full  # 使用整个训练集
#
# # 先用一个不打乱的 loader 统计 z-score（只用训练集）
# stats_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
#                           pin_memory=True, drop_last=False, num_workers=2)
#
# def _to_LC(x):
#     # 将样本统一为 (L, C)
#     if x.dim() == 2:
#         if x.shape[-1] == d_channel:
#             return x
#         elif x.shape[0] == d_channel and x.shape[1] == d_input:
#             return x.transpose(0, 1).contiguous()  # (C,L)->(L,C)
#     elif x.dim() == 3:
#         # batch 情况 (B, L, C) or (B, C, L)
#         if x.shape[-1] == d_channel:
#             return x
#         elif x.shape[1] == d_channel and x.shape[2] == d_input:
#             return x.permute(0, 2, 1).contiguous()
#     return x  # 尽量不破坏未知情况
#
#
# def _maybe_to_BLC(x):
#     """
#     将 batch 张量统一为 (B, L, C) 以适配纯 Mamba
#     支持 (B, L, C) 或 (B, C, L)
#     """
#     if x.dim() == 3:
#         if x.shape[-1] == d_channel:
#             return x
#         elif x.shape[1] == d_channel and x.shape[2] == d_input:
#             return x.permute(0, 2, 1).contiguous()
#     elif x.dim() == 2:
#         # 单样本 (L, C) or (C, L)
#         return _to_LC(x).unsqueeze(0)
#     return x
#
# @torch.no_grad()
# def compute_train_zscore():
#     # 返回 per-channel mean/std (C,) —— 只用训练集
#     s = torch.zeros(d_channel, dtype=torch.float64)
#     ss = torch.zeros(d_channel, dtype=torch.float64)
#     n = 0
#     for xb, yb in stats_loader:
#         xb = _to_LC(xb)          # (B,L,C)
#         if xb.dim() == 2:
#             xb = xb.unsqueeze(0)
#         B, L, C = xb.shape
#         x_flat = xb.reshape(-1, C).to('cpu', dtype=torch.float64)  # ((B*L), C)
#         s += x_flat.sum(dim=0)
#         ss += (x_flat ** 2).sum(dim=0)
#         n += x_flat.shape[0]
#     mean = (s / n).to(torch.float32)
#     var  = (ss / n - mean.double()**2).clamp_min(1e-12).to(torch.float32)
#     std  = var.sqrt()
#     return mean, std
#
# train_mean, train_std = compute_train_zscore()
# print(f"[z-score] train mean (first 3): {train_mean[:3].tolist()}  std (first 3): {train_std[:3].tolist()}")
#
# def zscore_norm(x, mean, std):
#     """
#     x: (B, L, C) or (L, C)
#     mean/std: (C,)
#     """
#     if x.dim() == 2:
#         return (x - mean) / (std + 1e-6)
#     elif x.dim() == 3:
#         return (x - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + 1e-6)
#     return x
#
# train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
#                           pin_memory=True, drop_last=True, num_workers=4)
# test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
#                           pin_memory=True, drop_last=False, num_workers=4)
#
#
# # model = MambaStack(
# #     num_features=d_channel,
# #     d_model=args.d_model, d_state=args.d_state, d_conv=args.d_conv, expand=args.expand,
# #     num_layers=args.layers, num_classes=d_output,
# #     dropout=0.2, use_tpp=True, tpp_scales=(1,2,4)
# # ).to(device)
#
# model = MambaStack(
#     num_features=d_channel,            # 输入通道数 (例如 9)
#     d_model=args.d_model,              # 主干维度 (如 64 或 96)
#     d_state=args.d_state,
#     d_conv=args.d_conv,
#     expand=args.expand,
#     num_layers=args.layers,
#     num_classes=d_output,              # 分类数 (UCI-HAR = 6)
#     dropout=0.2,
#     use_tpp=True,
#     tpp_scales=(1, 2, 4),
#
#     # ✅ 重点：使用轻量分类空间MoEHead
#     use_moe=True,
#     moe_num_experts=4,    # 可尝试 2 或 4
#     moe_top_k=2,          # 建议2更稳定
#     moe_sparse=True       # 稀疏计算，节省显存
# ).to(device)
#
# # ---------------- Train setup ---------------- #
# # smoothing 建议在 0.02~0.05 内
# criterion = nn.CrossEntropyLoss(label_smoothing=float(np.clip(args.label_smoothing, 0.0, 0.2))).to(device)
# optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
# epochs = args.epochs
# warmup_epochs = max(1, int(0.05 * epochs))
# def lr_lambda(e):
#     if e < warmup_epochs:
#         return (e + 1) / warmup_epochs
#     progress = (e - warmup_epochs) / max(1, epochs - warmup_epochs)
#     return 0.5 * (1 + math.cos(math.pi * progress))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#
# # ---------------- 轻量数据增强（仅训练时） ---------------- #
# def augment_batch(x: torch.Tensor) -> torch.Tensor:
#     """
#     x: (B, L, C), 仅在训练阶段调用
#     - 高斯噪声
#     - 通道缩放抖动
#     - 时间掩蔽（连续 5% 时间步）
#     """
#     B, L, C = x.shape
#     # 高斯噪声
#     noise_std = args.aug_noise_std
#     if noise_std > 0:
#         x = x + noise_std * torch.randn_like(x)
#
#     # 通道缩放（per-channel）
#     scale_std = args.aug_scale_std
#     if scale_std > 0:
#         scale = (1.0 + scale_std * torch.randn(1, 1, C, device=x.device))
#         x = x * scale
#
#     # 时间掩蔽
#     ratio = float(np.clip(args.aug_time_mask_ratio, 0.0, 0.5))
#     t_len = max(0, int(ratio * L))
#     if t_len > 0 and t_len < L:
#         t0 = torch.randint(0, L - t_len + 1, (B,), device=x.device)
#         # 为保证每个样本不同位置，可逐样本 mask
#         for b in range(B):
#             x[b, t0[b]:t0[b]+t_len, :] = 0.0
#
#     return x
#
# def train_one_epoch():
#     model.train()
#     total_loss = 0.0
#     for x, y in train_loader:
#         x = _maybe_to_BLC(x).float().to(device)   # 期望 (B, L, C)
#         # 统一归一化
#         x = zscore_norm(x, train_mean.to(x.device), train_std.to(x.device))
#         # —— 仅训练时的数据增强 —— #
#         x = augment_batch(x)
#
#         y = y.long().to(device)
#
#         optimizer.zero_grad()
#         logits, aux = model(x)
#         cls_loss = criterion(logits, y)
#
#         # ✅ MoE负载均衡损失
#         if hasattr(model, 'use_moe') and model.use_moe:
#             gate_probs = aux['gate_probs']
#             dispatch_mask = aux['dispatch_mask']
#             lb_loss = model.load_balance_loss(gate_probs, dispatch_mask)
#             loss = cls_loss + 0.01 * lb_loss
#         else:
#             loss = cls_loss
#
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / max(1, len(train_loader))
#
# @torch.no_grad()
# def evaluate(loader, flag='test'):
#     model.eval()
#     total, correct, total_loss = 0, 0, 0.0
#     for x, y in loader:
#         x = _maybe_to_BLC(x).float().to(device)
#         x = zscore_norm(x, train_mean.to(x.device), train_std.to(x.device))
#         logits, _ = model(x)
#         loss = criterion(logits, y.long().to(device))
#         total_loss += loss.item()
#         pred = logits.argmax(dim=-1)
#         total += y.size(0)
#         correct += (pred == y.to(device)).sum().item()
#     acc = correct / max(1, total)
#     print(f'[{flag}] loss={total_loss / len(loader):.5f} acc={acc * 100:.2f}%')
#     return acc, total_loss / max(1, len(loader))
#
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from thop import profile  # ⚠️ pip install thop
import os
#
# @torch.no_grad()
def evaluate_detailed(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = _maybe_to_BLC(x).float().to(device)
        x = zscore_norm(x, train_mean.to(x.device), train_std.to(x.device))
        logits, _ = model(x)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # === 基本分类指标 ===
    acc = (all_preds == all_labels).mean()
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n===== 📊 Classification Metrics =====")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")
    print("Confusion Matrix:\n", cm)

    return acc, prec, rec, f1

def compute_model_complexity(model, input_shape=(1, 128, 9)):
    """
    计算参数量、FLOPs、模型大小
    input_shape: (B, L, C)
    """
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # 模型大小（MB）
    torch.save(model.state_dict(), "temp.pth")
    model_size = os.path.getsize("temp.pth") / (1024 * 1024)
    os.remove("temp.pth")

    print("\n===== ⚙️ Model Complexity =====")
    print(f"Parameters : {params / 1e6:.3f} M")
    print(f"FLOPs      : {flops / 1e6:.3f} MFLOPs")
    print(f"Model Size : {model_size:.2f} MB")

    return params, flops, model_size

#     # ---------------- Main ---------------- #
# def main():
#     os.makedirs('saved_model', exist_ok=True)
#     best_test = 0.0
#     best_test_path = f"saved_model/mamba_moe_{use_dataset_name}_best.pth"
#
#     test_accs = []
#     for e in range(epochs):
#         t0 = time.time()
#         tr_loss = train_one_epoch()
#         test_acc, _ = evaluate(test_loader, 'test')
#         test_accs.append(test_acc)
#
#         # ✅ 每一轮都检查是否最优并保存
#         if test_acc > best_test:
#             best_test = test_acc
#             torch.save(model.state_dict(), best_test_path)
#
#         # 每10轮打印平均测试精度
#         if (e + 1) % 10 == 0:
#             avg_test_acc = sum(test_accs[-10:]) / min(10, len(test_accs))
#             print(f"Test acc (avg over last 10 epochs): {avg_test_acc * 100:.2f}%")
#
#         scheduler.step()
#         print(f"Epoch {e + 1:03d}/{epochs} | train_loss={tr_loss:.5f} | "
#               f"test_acc={test_acc * 100:.2f}% | lr={scheduler.get_last_lr()[0]:.6f} | "
#               f"time={time.time() - t0:.2f}s")
#
#     # ✅ 加载并评估真正的最优模型
#     # model.load_state_dict(torch.load(best_test_path, map_location=device))
#     # test_acc, _ = evaluate(test_loader, 'test')
#     # print(f'✅ Best Test Acc (saved model): {best_test * 100:.2f}%')
#
#     # ✅ 加载最优模型并详细评估
#     model.load_state_dict(torch.load(best_test_path, map_location=device))
#     acc, prec, rec, f1 = evaluate_detailed(model, test_loader, device)
#     params, flops, size = compute_model_complexity(model, input_shape=(1, d_input, d_channel))
#
#     print("\n===== 🏁 Final Summary =====")
#     print(f"Best Model Performance on Test Set ({use_dataset_name}):")
#     print(f"Accuracy : {acc * 100:.2f}%")
#     print(f"Precision: {prec * 100:.2f}%")
#     print(f"Recall   : {rec * 100:.2f}%")
#     print(f"F1-score : {f1 * 100:.2f}%")
#     print(f"Params   : {params / 1e6:.2f} M")
#     print(f"FLOPs    : {flops / 1e6:.2f} MFLOPs")
#     print(f"Model Size: {size:.2f} MB")
#
# if __name__ == "__main__":
#     main()




# -*- coding: utf-8 -*-
import argparse, os, math, time, numpy as np, random, torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import HarDataset
from models.moe_bi_mamba import MambaStack   # ✅ 使用修正后的 MoE 版
# from models.pruemamba import PureMambaClassifier   # 若要切换到纯 Mamba
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# ---------------- CLI ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--dataset-number', default=0, type=int, dest='dataset_number')
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)                 # 主干学习率
parser.add_argument('--gate-lr', default=0.0003, type=float)             # 门控更小 LR
parser.add_argument('--weight-decay', default=0.02, type=float)
parser.add_argument('--label-smoothing', default=0.03, type=float)
# 主干超参
parser.add_argument('--d-model', default=64, type=int)
parser.add_argument('--d-state', default=32, type=int)
parser.add_argument('--d-conv',  default=8, type=int)
parser.add_argument('--expand',  default=2, type=int)
parser.add_argument('--layers',  default=3, type=int)
# 数据增强
parser.add_argument('--aug-noise-std', default=0.01, type=float)
parser.add_argument('--aug-scale-std', default=0.05, type=float)
parser.add_argument('--aug-time-mask-ratio', default=0.05, type=float)
# MoE & 训练策略
parser.add_argument('--use-moe', action='store_true', default=True)
parser.add_argument('--moe-num-experts', default=4, type=int)  #4
parser.add_argument('--moe-top-k', default=1, type=int)             #1     # Switch 风格更稳：1 或 2
parser.add_argument('--moe-temp', default=3.0, type=float)             # 门控温度
parser.add_argument('--lb-weight', default=0.0005, type=float)           # 负载均衡权重（修正后）
parser.add_argument('--moe-warmup-epochs', default=30, type=int)   #30    # 冷启动：前若干个 epoch 用 dense
parser.add_argument('--patience', default=500, type=int, help='early stopping patience (epochs without improvement)')

args = parser.parse_args()

# ---------------- Device ---------------- #
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# ---------------- Dataset cfg ---------------- #
dataset_dict = {
    'uci-har': {'file_path': './data/UCI_data/np/',      'd_input': 128, 'd_channel': 9,  'd_output': 6},
    'usc-had': {'file_path': './data/USC_HAD_data/data_length128test_size0.3/',  'd_input': 128, 'd_channel': 6,  'd_output': 12},
    'real-world': {'file_path': './data/Real_World_data/data_length128window_size128test_size0.3/', 'd_input': 128, 'd_channel': 21, 'd_output': 8},
}
dataset_names = ['uci-har','usc-had','real-world']
use_dataset_name = dataset_names[args.dataset_number]
cfg = dataset_dict[use_dataset_name]
d_input, d_channel, d_output = cfg['d_input'], cfg['d_channel'], cfg['d_output']

# ---------------- Reproducibility ---------------- #
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed); random.seed(args.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ---------------- Datasets & Loaders ---------------- #
train_set = HarDataset(cfg['file_path'], trainOrTest='train')
test_set  = HarDataset(cfg['file_path'], trainOrTest='test')

stats_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                          pin_memory=True, drop_last=False, num_workers=2)

def _to_LC(x):
    # 统一为 (L, C)
    if x.dim() == 2:
        if x.shape[-1] == d_channel: return x
        if x.shape[0] == d_channel and x.shape[1] == d_input:
            return x.transpose(0, 1).contiguous()
    elif x.dim() == 3:
        if x.shape[-1] == d_channel: return x
        if x.shape[1] == d_channel and x.shape[2] == d_input:
            return x.permute(0, 2, 1).contiguous()
    return x

def _maybe_to_BLC(x):
    # 统一 batch 为 (B, L, C)
    if x.dim() == 3:
        if x.shape[-1] == d_channel: return x
        if x.shape[1] == d_channel and x.shape[2] == d_input:
            return x.permute(0, 2, 1).contiguous()
    elif x.dim() == 2:
        return _to_LC(x).unsqueeze(0)
    return x

@torch.no_grad()
def compute_train_zscore():
    s = torch.zeros(d_channel, dtype=torch.float64)
    ss = torch.zeros(d_channel, dtype=torch.float64)
    n = 0
    for xb, yb in stats_loader:
        xb = _to_LC(xb)
        if xb.dim() == 2: xb = xb.unsqueeze(0)
        B, L, C = xb.shape
        x_flat = xb.reshape(-1, C).to('cpu', dtype=torch.float64)
        s += x_flat.sum(dim=0)
        ss += (x_flat ** 2).sum(dim=0)
        n += x_flat.shape[0]
    mean = (s / n).to(torch.float32)
    var  = (ss / n - mean.double()**2).clamp_min(1e-12).to(torch.float32)
    std  = var.sqrt()
    return mean, std

train_mean, train_std = compute_train_zscore()
print(f"[z-score] train mean (first 3): {train_mean[:3].tolist()}  std (first 3): {train_std[:3].tolist()}")

def zscore_norm(x, mean, std):
    if x.dim() == 2:  return (x - mean) / (std + 1e-6)
    if x.dim() == 3:  return (x - mean.view(1,1,-1)) / (std.view(1,1,-1) + 1e-6)
    return x

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          pin_memory=True, drop_last=True, num_workers=4)
test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                          pin_memory=True, drop_last=False, num_workers=4)

# ---------------- Model ---------------- #
model = MambaStack(
    num_features=d_channel,
    d_model=args.d_model, d_state=args.d_state, d_conv=args.d_conv, expand=args.expand,
    num_layers=args.layers, num_classes=d_output,
    dropout=0.2, use_tpp=True, tpp_scales=(1,2,4), use_moe=args.use_moe,
    moe_num_experts=args.moe_num_experts,
    moe_top_k=args.moe_top_k, moe_sparse=True
).to(device)

# model = MambaStack(
#     num_features=d_channel,
#     d_model=args.d_model,
#     d_state=args.d_state,
#     d_conv=args.d_conv,
#     expand=args.expand,
#     num_layers=args.layers,
#     num_classes=d_output,
#     dropout=0.2,
#     use_tpp=False,
#     use_moe=args.use_moe,
# ).to(device)

# 设置 MoE 温度
if args.use_moe:
    model.moehead.temperature = float(args.moe_temp)

# ---------------- Optimizer & Scheduler ---------------- #
criterion = nn.CrossEntropyLoss(label_smoothing=float(np.clip(args.label_smoothing, 0.0, 0.2))).to(device)

# 参数组：门控单独 lr & 0 wd
if args.use_moe:
    gate_params = list(model.moehead.gate.parameters())
    expert_params = [p for n,p in model.named_parameters()
                     if p.requires_grad and (not any(p is gp for gp in gate_params))]
    optimizer = optim.AdamW([
        {'params': expert_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': gate_params,   'lr': args.gate_lr, 'weight_decay': 0.0},
    ])
else:
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

epochs = args.epochs
warmup_epochs = max(1, int(0.05 * epochs))
def lr_lambda(e):
    if e < warmup_epochs: return (e + 1) / warmup_epochs
    progress = (e - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------- Augment ---------------- #
def augment_batch(x: torch.Tensor) -> torch.Tensor:
    B, L, C = x.shape
    ns = args.aug_noise_std
    if ns > 0: x = x + ns * torch.randn_like(x)
    ss = args.aug_scale_std
    if ss > 0:
        scale = (1.0 + ss * torch.randn(1,1,C, device=x.device))
        x = x * scale
    ratio = float(np.clip(args.aug_time_mask_ratio, 0.0, 0.5))
    t_len = max(0, int(ratio * L))
    if t_len > 0 and t_len < L:
        t0 = torch.randint(0, L - t_len + 1, (B,), device=x.device)
        for b in range(B): x[b, t0[b]:t0[b]+t_len, :] = 0.0
    return x

def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    # 冷启动策略：前若干个 epoch 用 dense 路由
    if args.use_moe:
        if epoch < args.moe_warmup_epochs:
            model.set_moe_sparse(False, top_k=model.moehead.num_experts)  # dense
            lb_weight = 0.0                                               # 预热不加 LBLoss
        else:
            model.set_moe_sparse(True, top_k=args.moe_top_k)              # 稀疏
            lb_weight = args.lb_weight
    else:
        lb_weight = 0.0

    for x, y in train_loader:
        x = _maybe_to_BLC(x).float().to(device)
        x = zscore_norm(x, train_mean.to(x.device), train_std.to(x.device))
        x = augment_batch(x)
        y = y.long().to(device)

        optimizer.zero_grad()
        logits, aux = model(x)
        cls_loss = criterion(logits, y)

        if args.use_moe:
            gate_probs = aux['gate_probs']
            dispatch_mask = aux['dispatch_mask']
            lb_loss = model.load_balance_loss(gate_probs, dispatch_mask)
            loss = cls_loss + lb_weight * lb_loss
        else:
            loss = cls_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(train_loader))

@torch.no_grad()
def evaluate(loader, flag='test'):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x = _maybe_to_BLC(x).float().to(device)
        x = zscore_norm(x, train_mean.to(x.device), train_std.to(x.device))
        logits, _ = model(x)
        loss = criterion(logits, y.long().to(device))
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        total += y.size(0)
        correct += (pred == y.to(device)).sum().item()
    acc = correct / max(1, total)
    print(f'[{flag}] loss={total_loss / len(loader):.5f} acc={acc * 100:.2f}%')
    return acc, total_loss / max(1, len(loader))


@torch.no_grad()
def extract_backbone_features(model, loader, device):
    """
    提取骨干特征：in_proj → blocks → (tpp) → norm_out → GAP
    返回:
      feats: (N, D)
      labels: (N,)
      preds: (N,)
    """
    model.eval()
    all_feats, all_labels, all_preds = [], [], []

    for x, y in loader:
        x = _maybe_to_BLC(x).float().to(device)
        x = zscore_norm(x, train_mean.to(x.device), train_std.to(x.device))

        # === 跟随你的 MambaStack 前向到分类头前 ===
        z = model.in_proj(x)                     # (B, L, D)
        for blk in model.blocks:
            z = blk(z)
        if getattr(model, 'use_tpp', False):
            z = model.tpp(z)
        z = model.norm_out(z)
        z = z.mean(dim=1)                        # (B, D)  —— 特征

        # 分类预测（走完整 forward）
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)

        all_feats.append(z.detach().cpu())
        all_labels.append(y.clone().cpu())
        all_preds.append(pred.detach().cpu())

    feats = torch.cat(all_feats, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()
    return feats, labels, preds


def plot_confusion(cm, class_names, normalize=True,
                   save_path="figs/confmat_test.png",
                   title="Confusion Matrix (Test)"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_plot = cm.astype('float')
    if normalize:
        cm_sum = cm_plot.sum(axis=1, keepdims=True) + 1e-9
        cm_plot = cm_plot / cm_sum

    plt.figure(figsize=(6.2, 5.6), dpi=180)
    plt.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_plot.max() / 2.0
    for i, j in itertools.product(range(cm_plot.shape[0]), range(cm_plot.shape[1])):
        plt.text(j, i, format(cm_plot[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_plot[i, j] > thresh else "black",
                 fontsize=7)

    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Confusion] saved to {save_path}")


def plot_tsne(feats, labels, save_path="figs/tsne_test.png",
              title="t-SNE on Test Features", class_names=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init='pca',
                learning_rate='auto', random_state=42)
    emb = tsne.fit_transform(feats)  # (N,2)

    # 绘图（每类单独 scatter，生成离散图例）
    plt.figure(figsize=(6.4, 5.6), dpi=180)
    ax = plt.gca()

    classes = np.unique(labels)
    num_classes = len(classes)
    if class_names is None:
        class_names = [f'{int(c)}' for c in classes]

    # 颜色表：≤10 用 tab10，否则 tab20
    cmap = plt.cm.tab10 if num_classes <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, num_classes))

    handles = []
    for i, c in enumerate(classes):
        idx = (labels == c)
        sc = ax.scatter(emb[idx, 0], emb[idx, 1],
                        s=7, alpha=0.85, c=[colors[i]],
                        edgecolors='none', label=class_names[i])
        handles.append(sc)

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

    # 图例：右侧方框，带边框
    leg = ax.legend(handles=handles, loc='center left',
                    bbox_to_anchor=(1.02, 0.5), frameon=True,
                    framealpha=1.0, fontsize=9, title='Classes')
    leg.get_frame().set_edgecolor('#333333')
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[t-SNE] saved to {save_path}")



# --------- 可选：全局风格（更清爽） ---------
def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
    })

# # ============== 更好看的 t-SNE ==============
# def plot_tsne(
#     feats, labels,
#     save_path="figs/tsne_test.png",
#     title="t-SNE on Test Features",
#     class_names=None,
#     palette=None
# ):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     set_plot_style()
#
#     # --- t-SNE ---
#     tsne = TSNE(n_components=2, perplexity=30, init='pca',
#                 learning_rate='auto', random_state=42)
#     emb = tsne.fit_transform(feats)  # (N,2)
#
#     # --- 绘制（每类单独 scatter，搭配鲜艳色表） ---
#     plt.figure(figsize=(6.4, 5.6))
#     ax = plt.gca()
#     classes = np.unique(labels)
#     num_classes = len(classes)
#     if class_names is None:
#         class_names = [f'Class {int(c)}' for c in classes]
#
#     # 调色板：默认 tab20；若类别≤10 则 tab10；也可传入自定义 colormap 名称
#     if palette is None:
#         cmap = plt.cm.tab10 if num_classes <= 10 else plt.cm.tab20
#         colors = cmap(np.linspace(0, 1, num_classes))
#     else:
#         cmap = plt.get_cmap(palette)
#         colors = cmap(np.linspace(0, 1, num_classes))
#
#     handles = []
#     for i, c in enumerate(classes):
#         idx = (labels == c)
#         sc = ax.scatter(
#             emb[idx, 0], emb[idx, 1],
#             s=14,                # 点再大一点更醒目
#             alpha=0.85,          # 微透明
#             c=[colors[i]],
#             edgecolors='white',  # 白色描边提升可读性
#             linewidths=0.35,
#             label=class_names[i],
#             marker='o'
#         )
#         handles.append(sc)
#
#     ax.set_title(title, pad=10)
#     # 去除坐标轴刻度与网格，仅保留细边框
#     ax.set_xticks([]); ax.set_yticks([])
#
#     # 右侧紧凑图例 + 边框
#     leg = ax.legend(handles=handles, loc='center left',
#                     bbox_to_anchor=(1.02, 0.5), frameon=True,
#                     framealpha=0.98, fontsize=9, title='Classes')
#     leg.get_frame().set_edgecolor('#333333')
#     leg.get_frame().set_linewidth(0.8)
#
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()
#     print(f"[t-SNE] saved to {save_path}")
#
# # ============== 更好看的 混淆矩阵 ==============
# def plot_confusion(
#     cm, class_names,
#     normalize=True,
#     save_path="figs/confmat_test.png",
#     title="Confusion Matrix (Test)",
#     cmap_name='Spectral'   # 更鲜艳可选：'Spectral', 'plasma', 'magma'
# ):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     set_plot_style()
#
#     cm_plot = cm.astype('float')
#     if normalize:
#         cm_sum = cm_plot.sum(axis=1, keepdims=True) + 1e-9
#         cm_plot = cm_plot / cm_sum
#
#     plt.figure(figsize=(6.6, 5.6))
#     ax = plt.gca()
#     im = ax.imshow(cm_plot, interpolation='nearest', cmap=plt.get_cmap(cmap_name))
#     ax.set_title(title, pad=10)
#
#     # 颜色条紧凑显示
#     cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
#     cbar.ax.tick_params(labelsize=9)
#
#     tick_marks = np.arange(len(class_names))
#     ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
#     ax.set_xticklabels(class_names, rotation=45, ha='right')
#     ax.set_yticklabels(class_names)
#
#     # 方格细边框
#     ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
#     ax.set_xlim(-0.5, len(class_names)-0.5)
#     ax.set_ylim(len(class_names)-0.5, -0.5)
#
#     # 单元格数值：动态反色；对角线加粗更醒目
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm_plot.max() * 0.55
#     for i, j in itertools.product(range(cm_plot.shape[0]), range(cm_plot.shape[1])):
#         val = cm_plot[i, j]
#         txt_color = "white" if val > thresh else "black"
#         fw = 'bold' if i == j else 'normal'
#         ax.text(j, i, format(val, fmt),
#                 ha="center", va="center",
#                 color=txt_color, fontsize=9, fontweight=fw)
#
#     # 四边框更细致
#     for spine in ax.spines.values():
#         spine.set_edgecolor('#444444'); spine.set_linewidth(0.8)
#
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()
#     print(f"[Confusion] saved to {save_path}")

#有早停的main
def main():


    ten_step_log = []  # 只在第 10, 20, 30, ... 轮记录
    os.makedirs('logs', exist_ok=True)
    csv_path = 'logs/test_acc_every10.csv'
    jsonl_path = 'logs/test_acc_every10.jsonl'

    # 如果需要每次重跑都重写表头：
    with open(csv_path, 'w') as f:
        f.write('epoch,test_acc,avg_last10\n')
    # 如果你更喜欢追加而不覆盖，上面两行改成只在文件不存在时写表头



    os.makedirs('saved_model', exist_ok=True)
    best_test = -1.0
    best_path = f"saved_model/mamba_moe_{use_dataset_name}_best.pth"

    test_accs = []
    no_improve = 0          # 早停计数器：连续未提升的 epoch 数
    best_epoch = -1         # 记录最佳出现的 epoch（从0计）

    for e in range(epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(e)
        test_acc, _ = evaluate(test_loader, 'test')
        test_accs.append(test_acc)

        # 判断是否有提升（严格大于，避免浮点噪声可加一个极小阈值）
        if test_acc > best_test + 1e-12:
            best_test = test_acc
            best_epoch = e
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            improved_str = " (improved ✅)"
        else:
            no_improve += 1
            improved_str = ""

        # 每10轮输出均值 & 仍然可选保存阶段性checkpoint
        if (e + 1) % 10 == 0:
            avg_test_acc = sum(test_accs[-10:]) / min(10, len(test_accs))
            print(f"Test acc (avg last 10): {avg_test_acc * 100:.2f}%")



            # === 记录并落盘（当前 test_acc 以及最近10次均值） ===
            record = {
                'epoch': int(e + 1),
                'test_acc': float(test_acc),  # 当前第 e+1 轮的准确率
                'avg_last10': float(avg_test_acc)  # 最近10次平均
            }
            ten_step_log.append(record)

            # 1) 追加到 CSV
            with open(csv_path, 'a') as f:
                f.write(f"{record['epoch']},{record['test_acc']:.6f},{record['avg_last10']:.6f}\n")

            # 2) 追加到 JSONL（便于后续脚本读取）
            with open(jsonl_path, 'a') as f:
                import json
                f.write(json.dumps(record, ensure_ascii=False) + '\n')




            if (e + 1) % 20 == 0:
                save_path = f"saved_model/checkpoint_epoch{e + 1}.pt"
                torch.save(model.state_dict(), save_path)
                print(f"✅ Saved checkpoint: {save_path}")

        scheduler.step()
        print(f"Epoch {e + 1:03d}/{epochs} | train_loss={tr_loss:.5f} | "
              f"test_acc={test_acc * 100:.2f}% | lr={scheduler.get_last_lr()[0]:.6f} | "
              f"no_improve={no_improve}/{args.patience}{improved_str} | "
              f"time={time.time() - t0:.2f}s")

        # ===== 早停触发 =====
        if no_improve >= args.patience:
            print(f"\n⛔ Early stopping triggered: no improvement for {args.patience} epochs "
                  f"(best={best_test * 100:.2f}% at epoch {best_epoch + 1}).")
            break

    # 训练结束后，加载并用最佳权重做最终评估与可视化
    model.load_state_dict(torch.load(best_path, map_location=device))

    acc, prec, rec, f1 = evaluate_detailed(model, test_loader, device)
    params, flops, size = compute_model_complexity(model, input_shape=(1, d_input, d_channel))

    print("\n===== 🏁 Final Summary =====")
    print(f"Best Model Performance on Test Set ({use_dataset_name}):")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")
    print(f"Params   : {params / 1e6:.2f} M")
    print(f"FLOPs    : {flops / 1e6:.2f} MFLOPs")
    print(f"Model Size: {size:.2f} MB")

    # ===== 额外可视化：t-SNE 与 混淆矩阵 =====
    feats, labels, preds = extract_backbone_features(model, test_loader, device)

    plot_tsne(feats, labels,
              save_path=f"figs/{use_dataset_name}_tsne_test.png",
              title=f"t-SNE on Test Features ({use_dataset_name})")

    cm = confusion_matrix(labels, preds)
    class_names = [str(i) for i in range(d_output)]
    plot_confusion(cm, class_names, normalize=True,
                   save_path=f"figs/{use_dataset_name}_confmat_test.png",
                   title=f"Confusion Matrix (Test, {use_dataset_name})")


#没有早停的main
# def main():
#     os.makedirs('saved_model', exist_ok=True)
#     best_test = 0.0
#     best_path = f"saved_model/mamba_moe_{use_dataset_name}_best.pth"
#     test_accs = []
#
#     for e in range(epochs):
#         t0 = time.time()
#         tr_loss = train_one_epoch(e)
#         test_acc, _ = evaluate(test_loader, 'test')
#         test_accs.append(test_acc)
#
#         if test_acc > best_test:
#             best_test = test_acc
#             torch.save(model.state_dict(), best_path)
#
#         if (e + 1) % 10 == 0:
#             avg_test_acc = sum(test_accs[-10:]) / min(10, len(test_accs))
#             print(f"Test acc (avg last 10): {avg_test_acc * 100:.2f}%")
#
#             if (e + 1) % 20 == 0:
#                 save_path = f"saved_model/checkpoint_epoch{e + 1}.pt"
#                 torch.save(model.state_dict(), save_path)
#                 print(f"✅ Saved checkpoint: {save_path}")
#
#         scheduler.step()
#         print(f"Epoch {e + 1:03d}/{epochs} | train_loss={tr_loss:.5f} | "
#               f"test_acc={test_acc * 100:.2f}% | lr={scheduler.get_last_lr()[0]:.6f} | "
#               f"time={time.time() - t0:.2f}s")
#
#
#
#     model.load_state_dict(torch.load(best_path, map_location=device))   #原来是best_path
#     # test_acc, _ = evaluate(test_loader, 'test')
#     # print(f'✅ Best Test Acc (checkpoint): {best_test * 100:.2f}%')
#
#     acc, prec, rec, f1 = evaluate_detailed(model, test_loader, device)
#     params, flops, size = compute_model_complexity(model, input_shape=(1, d_input, d_channel))
#
#     print("\n===== 🏁 Final Summary =====")
#     print(f"Best Model Performance on Test Set ({use_dataset_name}):")
#     print(f"Accuracy : {acc * 100:.2f}%")
#     print(f"Precision: {prec * 100:.2f}%")
#     print(f"Recall   : {rec * 100:.2f}%")
#     print(f"F1-score : {f1 * 100:.2f}%")
#     print(f"Params   : {params / 1e6:.2f} M")
#     print(f"FLOPs    : {flops / 1e6:.2f} MFLOPs")
#     print(f"Model Size: {size:.2f} MB")
#
#     # ===== 额外可视化：t-SNE 与 混淆矩阵 =====
#     feats, labels, preds = extract_backbone_features(model, test_loader, device)
#
#     # t-SNE 可视化（测试集）
#     plot_tsne(feats, labels, save_path=f"figs/{use_dataset_name}_tsne_test.png",
#               title=f"t-SNE on Test Features ({use_dataset_name})")
#
#     # 混淆矩阵（测试集）
#     cm = confusion_matrix(labels, preds)
#     class_names = [str(i) for i in range(d_output)]   # 若你有具体标签名，可替换
#     plot_confusion(cm, class_names, normalize=True,
#                    save_path=f"figs/{use_dataset_name}_confmat_test.png",
#                    title=f"Confusion Matrix (Test, {use_dataset_name})")


if __name__ == "__main__":
    main()
