import torch
import torch.nn as nn
import torch.optim as optim
from model import Net
from pre_a2 import train_loader, val_loader, n_train, n_val, test_loader, n_test
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

plt.switch_backend('agg')
device = torch.device("cuda:1")
torch.cuda.set_per_process_memory_fraction(0.5, 1)
net = Net().to(device)
Epoch = 180
optimizer = optim.Adam(net.parameters(), lr=1e-4)

loss_func = nn.CrossEntropyLoss()

train_loss_all = []
val_loss_all = []
train_acc_all = []
val_acc_all = []


for epoch in tqdm(range(Epoch)):
    net.train()
    train_loss = 0.0
    train_acc_num = 0
    for i, (e_name,p_name,seq, label) in enumerate(train_loader, 0):
        seq = seq.to(device)
        label = label.to(device)
        pred = net(seq)
        pred_lab = torch.argmax(pred, 1)
        loss = loss_func(pred, label)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc_num += torch.sum(pred_lab == label.data).cpu()
    train_loss_all.append(train_loss)
    train_acc_all.append(train_acc_num / n_train)

    net.eval()
    val_loss = 0.0
    val_acc_num = 0
    with torch.no_grad():
        for i, (e_name, p_name, seq, label) in enumerate(val_loader, 0):
            seq = seq.to(device)
            label = label.to(device)
            pred = net(seq)
            pred_lab = torch.argmax(pred, 1)
            loss = loss_func(pred, label)
            val_loss += loss.item()
            val_acc_num += torch.sum(pred_lab == label.data).cpu()
    val_loss_all.append(val_loss)
    val_acc_all.append(val_acc_num / n_val)

torch.save(net, 'model_a2.pkl')

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss_all, 'ro-', label="Train Loss")
plt.plot(val_loss_all, 'bs-', label="Val Loss")
plt.legend()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_acc_all, 'ro-', label="Train acc")
plt.plot(val_acc_all, 'bs-', label="Val acc")
plt.legend()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("ACC")

plt.savefig("../result/Loss_Acc_Roc/loss_acc_a2.png")
p_1 = []
m_label = []
net.eval()
with torch.no_grad():
    for i, (e_name, p_name, seq, label) in enumerate(test_loader):
        seq = seq.to(device)
        out = net(seq).cpu()
        res = torch.nn.functional.softmax(out, dim=1).detach().numpy().tolist()
        for mm in res:
            p_1.append(mm[1])
        m_label = m_label + label.detach().numpy().tolist()
fpr, tpr, threshold = roc_curve(m_label, p_1)
roc_auc = auc(fpr, tpr)
np.savetxt('../result/fpr_and_tpr/a2_fpr.csv', fpr, delimiter=',')
np.savetxt('../result/fpr_and_tpr/a2_tpr.csv', tpr, delimiter=',')

lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('auRoc Curve of cotton of a2')
plt.legend(loc="lower right")
plt.savefig("../result/Loss_Acc_Roc/roc_a2.png")
