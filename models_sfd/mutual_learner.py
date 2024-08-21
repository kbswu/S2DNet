import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_conv0 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.joint_projection_1 = nn.Conv2d(16 + 1, 8, kernel_size=1)
        self.joint_projection_2 = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, feat, y):
        feat = F.relu(self.t_conv0(feat))
        feat = F.relu(self.t_conv1(feat))
        feat = F.relu(self.t_conv2(feat))
        feat = self.t_conv3(feat)
        feat = torch.cat((feat, y), dim=1)
        feat = F.relu(self.joint_projection_1(feat))
        feat = self.joint_projection_2(feat)
        return feat


class MIDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(512, 256, kernel_size=1)
        self.c1 = nn.Conv2d(256, 128, kernel_size=1)
        self.c2 = nn.Conv2d(128, 64, kernel_size=1)
        self.c3 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        h = F.relu(self.c2(h))
        return self.c3(h)


class MILossGlobal(nn.Module):
    def __init__(self):
        super().__init__()
        self.mi_discriminator = MIDiscriminator()
        self.beta = 0.1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def compute_dis_loss(self, shape):
        shape_pred = self.discriminator(self.pool(shape).squeeze())
        return -torch.log(shape_pred).mean()


    def forward(self, shape, app):
        shape_pred = self.discriminator(self.pool(shape).squeeze())
        app_pred = self.discriminator(self.pool(app).squeeze())
        loss = -torch.log(shape_pred) - torch.log(1 - app_pred)
        return loss.mean() * self.beta


class MILossLocal(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mi_discriminator = GlobalDiscriminator()
        self.beta = 0.1
        self.num_classes = num_classes

    def forward(self, shape, app, y):
        shape_prime = torch.cat((shape[1:], shape[0].unsqueeze(0)), dim=0)
        app_prime = torch.cat((app[1:], app[0].unsqueeze(0)), dim=0)
        if self.num_classes == 2:
            y = y[:, 1, :, :].unsqueeze(1)
        else:
            y = y[:, 2, :, :].unsqueeze(1)
        Ej_shape = -F.softplus(-self.mi_discriminator(shape, y)).mean()
        Em_shape = F.softplus(self.mi_discriminator(shape_prime, y)).mean()
        Ej_app = -F.softplus(-self.mi_discriminator(app, y)).mean()
        Em_app = F.softplus(self.mi_discriminator(app_prime, y)).mean()
        loss = (Em_shape - Ej_shape + Em_app - Ej_app) * self.beta
        return loss * self.beta


class MutualLearningModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.global_discriminator = MILossGlobal()
        self.local_discriminator = MILossLocal(num_classes)

    def forward(self, y, feats):
        shape, app = torch.chunk(feats, 2, dim=1)
        loss_global = self.global_discriminator(shape, app)
        loss_local = self.local_discriminator(shape, app, y)
        return loss_global + loss_local


if __name__ == "__main__":
    feat = torch.randn(2, 1024, 16, 16)
    pred = torch.randn(2, 2, 256, 256)
    model = MutualLearningModel(2)
    loss = model(pred, feat)
    print(loss)

