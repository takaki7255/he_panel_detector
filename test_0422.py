import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np

# データセットのパス
# TODO: 画像とアノテーションのパスを指定
image_paths = ""
annotation_paths = ""

# デバイスの設定 m1mac用gpuならmpsでcpuならcpu
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# ResNet101をバックボーンとして使用
backbone = torchvision.models.resnet101(pretrained=True).to(device)
backbone.out_channels = 2048

# アンカーボックスのサイズと縦横比を設定
anchor_sizes = ((128,), (256,), (512,))
aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)

# RPNのアンカージェネレータを生成
anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

# ROI Poolingの出力サイズを設定
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

# Faster R-CNNモデルを定義
model = FasterRCNN(
    backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler  # 背景とパネルの2クラス
).to(device)


# 2段階目のモデル
class SecondStageModel(nn.Module):
    def __init__(self, num_classes, roi_output_size=7):
        super(SecondStageModel, self).__init__()
        self.fc1 = nn.Linear(2048 * roi_output_size * roi_output_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 18)  # 四角形の頂点座標(x,y)×4=8次元

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred


# 2段階目のモデルを初期化
second_stage_model = SecondStageModel(num_classes=2).to(device)


# 2段階目の損失関数
def second_stage_loss(cls_score, bbox_pred, proposals, targets):
    # 正解ラベルの作成
    labels = torch.zeros(len(proposals)).long().to(device)
    for i, target in enumerate(targets):
        iou = box_iou(proposals[i], target["boxes"])
        labels[i] = (iou >= 0.5).long().clamp(0, 1)

    # 分類損失
    cls_loss = F.cross_entropy(cls_score, labels)

    # 回帰損失 (正解ラベルが1の場合のみ計算)
    reg_targets = [target["boxes"] for target in targets]
    reg_loss = quadrilateral_regression_loss(bbox_pred[labels == 1], reg_targets[labels == 1])

    return {"cls_loss": cls_loss, "reg_loss": reg_loss}


# 四角形回帰の損失関数
def quadrilateral_regression_loss(pred, target, sigma=1.0, lambda_=1 / 9):
    pred_values = pred[:, :10]
    target_values = target[:, :10]

    pred_signs = pred[:, 10:]
    target_signs = target[:, 10:]

    # Lvalue (Smooth L1 loss)
    diff = torch.abs(pred_values - target_values)
    loss_value = torch.where(diff < lambda_ / sigma, 0.5 * sigma**2 * diff**2, sigma * diff - 0.5 * lambda_)
    loss_value = loss_value.sum(dim=1).mean()

    # Lsign (Cross entropy loss)
    loss_sign = F.binary_cross_entropy_with_logits(pred_signs, target_signs, reduction="none")
    loss_sign = loss_sign.sum(dim=1).mean()

    # Total loss
    loss = loss_value + loss_sign
    return loss


# データセットの定義
class ComicDataset(Dataset):
    def __init__(self, image_paths, annotation_paths):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(annotation_path, "r") as f:
            annotations = f.readlines()

        boxes = []
        for annotation in annotations:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, annotation.split(","))
            boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        return image, target


# データローダーの作成
def collate_fn(batch):
    return tuple(zip(*batch))


dataset = ComicDataset(image_paths, annotation_paths)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 損失関数と最適化アルゴリズムを設定
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

second_stage_optimizer = torch.optim.SGD(second_stage_model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
second_stage_lr_scheduler = torch.optim.lr_scheduler.StepLR(second_stage_optimizer, step_size=3, gamma=0.1)

# 学習ループ
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    second_stage_model.train()

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 1段階目
        features = model.backbone(images)
        proposals, _ = model.rpn(images, features, targets)

        # 2段階目
        proposal_features = roi_pooler(features, proposals)
        cls_score, bbox_pred = second_stage_model(proposal_features)

        second_stage_loss_dict = second_stage_loss(cls_score, bbox_pred, proposals, targets)
        second_stage_losses = sum(loss for loss in second_stage_loss_dict.values())

        second_stage_optimizer.zero_grad()
        second_stage_losses.backward()
        second_stage_optimizer.step()

    second_stage_lr_scheduler.step()

print("Training completed.")
