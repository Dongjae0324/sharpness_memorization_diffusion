import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import torch

#TODO: manually insert path for comparison
mem_data, nmem_data = torch.load('./det_outputs/sd1_mem_gen4.pt', weights_only=True),\
                        torch.load('./det_outputs/sd1_nmem_gen4.pt', weights_only=True)

mem_data, nmem_data = mem_data.mean(dim=1).float().numpy(), nmem_data.mean(dim=1).float().numpy()
scores, labels = np.concatenate([mem_data, nmem_data]), np.concatenate([np.ones(mem_data.shape[0]), np.zeros(nmem_data.shape[0])])
auroc = roc_auc_score(labels, scores)

if auroc < 0.5:
    #change the labels if lower than 0.5
    auroc = auroc if auroc > 0.5 else 1 - auroc
    labels = 1 - labels

fpr, tpr, thresholds = roc_curve(labels, scores)

tpr_lst = []
thre_fpr = [0.01, 0.03]  
for thres in thre_fpr:
    target_fpr = thres
    closest_fpr_index = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_target_fpr = tpr[closest_fpr_index]
    tpr_lst.append(tpr_at_target_fpr)
    
print(f'AUC: {auroc:.3f} | TPR@1%FPR: {tpr_lst[0]:.3f} | TPR@3%FPR: {tpr_lst[1]:.3f}')
    
