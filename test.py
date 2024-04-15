import torch
import torch.nn as nn
import torchvision

# ResNet101をベースネットワークとして使用
backbone = torchvision.models.resnet101(pretrained=True)


class QuadrilateralRegressionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 回帰ターゲットを計算するための全結合層
        self.fc_reg = nn.Linear(1024, 18)

    def forward(self, x):
        # 入力特徴量から回帰ターゲットを予測
        x = self.fc_reg(x)
        return x


class PanelProposalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone.features[:7]  # ResNetの前半部分を使用

        # RPN層 (anchor生成、objectness分類、quadrilateral regression)
        self.rpn_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.rpn_cls = nn.Conv2d(512, 18, 1, 1, 0)
        self.rpn_reg = QuadrilateralRegressionLayer()

    def forward(self, x):
        x = self.backbone(x)
        x = self.rpn_conv(x)

        # objectness分類スコア
        rpn_cls_score = self.rpn_cls(x)

        # quadrilateral regression
        rpn_reg_pred = self.rpn_reg(x)

        return rpn_cls_score, rpn_reg_pred


class PanelDetectionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone  # ResNet101全体を使用
        self.roi_align = torchvision.ops.RoIAlign(7, 1.0)

        # 分類層とquadrilateral regression層
        self.fc_cls = nn.Linear(2048, 2)
        self.fc_reg = QuadrilateralRegressionLayer()

    def forward(self, x, proposals):
        x = self.backbone(x)

        # RoIAlign適用
        pool = self.roi_align(x, proposals)

        # 分類スコア出力
        cls_score = self.fc_cls(pool.view(-1, 2048))

        # quadrilateral regression
        reg_pred = self.fc_reg(pool.view(-1, 2048))

        return cls_score, reg_pred


# 全体のモデル
class QuadrilateralRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ppn = PanelProposalNetwork()
        self.pdn = PanelDetectionNetwork()

    def forward(self, x):
        # Stage 1: Panel Proposal Network
        rpn_cls, rpn_reg = self.ppn(x)

        # NMSでパネル候補を選択
        proposals = torchvision.nms(rpn_cls, rpn_reg)

        # Stage 2: Panel Detection Network
        cls_score, reg_pred = self.pdn(x, proposals)

        return cls_score, reg_pred
