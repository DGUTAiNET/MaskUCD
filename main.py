from datetime import datetime
import time
import math
import matplotlib.pyplot as plt
import csv
import torch.optim.lr_scheduler
from kits import *
from dataset import *
from model import MaskUCD


def experiment(
        dataset: str,
        epochs: int = 150,
        device=torch.device("cuda:0"),
        k_base :float = 2.0,
):
    _t1, _t2, _GT = DATASETS[dataset]()
    ch1, H, W = _t1.shape
    ch2 = _t2.shape[0]
    _t1 = _t1.unsqueeze(0).to(device)
    _t2 = _t2.unsqueeze(0).to(device)
    _GT = _GT.squeeze().cpu().numpy()
    time_str = datetime.now().strftime("%m%d%H%M%S")
    dir_path = f'Crop/{dataset}-{time_str}'
    make_dir(dir_path)

    metrics_history = {
        'epoch': [],
        "OA": [],
        "Precision": [],
        "Recall": [],
        "Kappa": [],
        "F1": [],
        "AUC": []
    }

    model = MaskUCD(in_ch1=ch1, in_ch2=ch2).to(device)
    MSE = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

    model.train()
    align_epoch = 50
    start_time = time.time()
    for epoch in range(epochs):
        alpha = 0.0 + (1 + math.cos(math.pi * epoch / align_epoch)) / 2 if epoch < align_epoch else 0.0

        t1_feat, t2_feat, pre_t1, pre_t2, pred_c_t1, pred_c_t2 = model(_t1, _t2)

        d = torch.norm(t2_feat - t1_feat, p=2, dim=1, keepdim=True)
        d = gaussian_blur_(d, kernel_size=5)
        d = (d-d.mean())/d.std()
        d = (d - d.min()) / (d.max() - d.min())
        k = k_base * (1 - alpha * 0.7)
        thre = d.mean() + k*d.std()
        cm = (d > thre).float().detach()
        # print(cm.sum()/cm.numel())
        d_np = d.squeeze().detach().cpu().numpy()
        cm_np = cm.squeeze().cpu().detach().numpy()

        if ((epoch + 1) % 20 == 0) or (epoch == 0):
            plt.figure(figsize=(15, 4))

            # heatmap
            ax1 = plt.subplot(1, 3, 1)
            im1 = ax1.imshow(d_np, cmap='viridis')
            ax1.set_title("Difference map (d)")
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # change map
            ax2 = plt.subplot(1, 3, 2)
            im2 = ax2.imshow(cm_np, cmap='gray')
            ax2.set_title("Binary Change Map (cm)")
            ax2.axis('off')

            # difference map
            ax3 = plt.subplot(1, 3, 3)
            ax3.hist(d_np.flatten(), bins=100, color='steelblue', alpha=0.8, edgecolor='black')
            ax3.set_title(f"thre={thre:.4f},mean={d.mean():.4f}")
            ax3.set_xlabel("d value")
            ax3.set_ylabel("Pixel count")
            ax3.grid(True)

            plt.tight_layout()
            plt.show()

        recon_loss1 = MSE(pre_t1, _t1)
        recon_loss2 = MSE(pre_t2, _t2)
        cross_loss = MSE(pred_c_t1*(1-cm), _t1*(1-cm)) + MSE(pred_c_t2*(1-cm), _t2*(1-cm))
        align_loss = MSE(t1_feat, t2_feat)
        loss_feat = (torch.exp(-d) * cm).sum() / (cm.sum())
        loss_unchanged = MSE(t1_feat * (1 - cm), t2_feat * (1 - cm))
        total_loss = recon_loss1 + recon_loss2 + (1-alpha)*loss_unchanged + loss_feat + alpha*align_loss + (1-alpha)*cross_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        transforms.ToPILImage()(cm.squeeze()).save(dir_path + f'/cm{epoch}.png')
        transforms.ToPILImage()(d.squeeze()).save(dir_path + f'/d{epoch}.png')
        print(f'Epoch: {epoch}, Total Loss: {total_loss.item():.4f}')
        print(
            f'recon_loss1:{recon_loss1:.4f},recon_loss2:{recon_loss2:.4f}'
            f'align_loss:{align_loss:.4f},loss_unchanged:{loss_unchanged:.4f},loss_feat:{loss_feat:.4f}， '
            f'cross_loss:{cross_loss:.4f}')

        metrics_history['epoch'].append(epoch)
        compute_metrics(_GT, cm_np, metrics_history)
        auc = calc_auc(d_np, _GT)
        metrics_history["AUC"].append(auc)
        print(f'AUC: {auc:.4f}')
    total_time = time.time() - start_time
    print(f"Running Time: {total_time:.2f} s")
    with open(f'{dir_path}/result.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=metrics_history.keys())
        writer.writeheader()
        for i in range(len(metrics_history['Kappa'])):
            row = {key: metrics_history[key][i] for key in metrics_history.keys()}
            writer.writerow(row)

    best_Kappa_index = metrics_history['Kappa'].index(max(metrics_history['Kappa']))
    best_score = {key: value[best_Kappa_index] for key, value in metrics_history.items()}
    last_score = {key: value[-1] for key, value in metrics_history.items()}
    print(f"Dataset:{dataset}\nBest result: {best_score}\nLast result: {last_score}")
    best_cm, metrics_text = bestCM(d_np, _GT)
    print(metrics_text)
    plt.imsave(dir_path + f'/cm-best.png', best_cm, cmap='gray', vmin=0, vmax=1)
    with open(f'{dir_path}/result.txt', 'w', encoding='utf-8') as file:
        file.write(f"Dataset:{dataset}\nBest result: {best_score}\nLast result: {last_score}")
        file.write('\nBest CM for last d:\n')
        file.write(metrics_text)
        file.write(f"Running Time: {total_time:.2f} s")


if __name__ == "__main__":
    dataset = [
        # 'Italy', # k_base = 2.0
        # 'Gloucester',  # k_base = 1.2
        # 'California', # k_base = 2.0
        # 'Texas', # k_base = 1.5
        # 'Toulouse', # k_base = 1.3
        # "Ottawa" # k_base = 1.3
        # 'YR1', # k_base = 2.5
        # "YR2"  # k_base = 3.3
        # 'YR3',  # k_base = 1.0
        # 'YR4', # k_base = 2.5
    ]
    experiment('Italy', k_base=2.0)
