import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, in_ch=3, img=64, num_classes=21, pool = 'avg'):
        super().__init__()

        if pool == 'avg': 
            Pool = nn.AvgPool2d 
        elif pool == 'max': 
            Pool = nn.MaxPool2d
        else: 
            raise ValueError("pool phải là 'avg' hoặc 'max'")

        self.pool_name = pool 

        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.s2 = Pool(2, 2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = Pool(2, 2)

        n_flat = self._flat_dim(in_ch, img)
        self.f5 = nn.Linear(n_flat, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

    def _flat_dim(self, in_ch, img):
        # Dò số chiều sau S4 bằng một forward giả, tránh tính tay
        with torch.no_grad():
            z = torch.zeros(1, in_ch, img, img)
            z = self.s4(self.c3(self.s2(self.c1(z))))
        return int(z.flatten(1).shape[1])

    def forward(self, x):
        x = self.s2(torch.relu(self.c1(x)))
        x = self.s4(torch.relu(self.c3(x)))
        x = x.flatten(1)
        x = torch.relu(self.f5(x))
        x = torch.relu(self.f6(x))
        return self.out(x)


def n_params(module):
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    for pool in ["avg", "max"]:
        model = LeNet5(
            in_ch=3,
            img=64,
            num_classes=21,
            pool=pool
        )

        x = torch.rand(2, 3, 64, 64)
        logits = model(x)

        print(f"\nPooling: {pool}")
        print("input :", tuple(x.shape))
        print("logits:", tuple(logits.shape))
        print("params:", f"{n_params(model):,}")
