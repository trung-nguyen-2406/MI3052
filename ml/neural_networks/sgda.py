import torch
from torch.optim import Optimizer

class SGDA(Optimizer):
    def __init__(self, params, lr=0.1, kappa=0.5, sigma=0.5):
        """
        Thuat toan SGDA nhu trong bai bao (Algorithm 3).
        lr: lambda_0 (buoc nhay khoi tao)
        kappa: he so giam buoc nhay khi khong thoa man dieu kien (0 < kappa < 1)
        sigma: tham so trong dieu kien Armijo (0 < sigma < 1)
        """
        defaults = dict(lr=lr, kappa=kappa, sigma=sigma)
        super(SGDA, self).__init__(params, defaults)

    def step(self, closure):
        """
        Thuc hien mot buoc cap nhat tham so.
        closure: ham de tinh lai loss (bat buoc phai co doi voi thuat toan nay)
        """
        loss = None
        # Tinh toan loss va gradient tai buoc hien tai (x^k)
        # Luu y: luc nay gradient da duoc tinh san boi loss.backward() o ben ngoai
        with torch.enable_grad():
            loss = closure()

        # Duyet qua cac nhom tham so (thuong chi co 1 nhom)
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            
            # 1. Thu thap tat ca gradient va tham so
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

            # 2. Tinh tong binh phuong norm cua gradient: ||grad||^2
            # Dieu kien: f(x_new) <= f(x) - sigma * lambda * ||grad||^2
            grad_norm_sq = 0.0
            for g in grads:
                grad_norm_sq += torch.sum(g * g).item()

            current_lr = group['lr']
            sigma = group['sigma']
            kappa = group['kappa']

            # 3. Cap nhat tham so: x^{k+1} = x^k - lambda_k * grad
            # (Luu lai trang thai cu de phong truong hop can dung, nhung thuat toan nay chap nhan buoc nhay luon)
            with torch.no_grad():
                for i, p in enumerate(params_with_grad):
                    p.add_(grads[i], alpha=-current_lr)

            # 4. Tinh loss tai diem moi: f(x^{k+1})
            # Can goi closure() mot lan nua
            with torch.enable_grad():
                new_loss = closure()

            # 5. Kiem tra dieu kien de cap nhat Learning Rate cho vong lap SAU (Algorithm 3 Step 1)
            # Dieu kien: f(x^{k+1}) <= f(x^k) - sigma * lambda_k * ||grad||^2
            # Luu y: term ben phai la "giam di mot luong", nen dau tru la dung.
            
            lhs = new_loss.item()
            rhs = loss.item() - sigma * current_lr * grad_norm_sq

            # Neu khong thoa man dieu kien -> Giam Learning Rate
            if lhs > rhs:
                group['lr'] = current_lr * kappa
                # print(f"  [SGDA] Giam LR xuong: {group['lr']:.6f}")
            else:
                # Giu nguyen LR
                pass
                
        return loss