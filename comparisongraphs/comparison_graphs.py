# comparison_graphs.py
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

matplotlib.use('Agg')

# ---------- Load Real GNN Results ----------
gnn_data = pd.read_csv("results/gnn_results.csv").iloc[0]

# ---------- Baseline Results (for comparison) ----------
models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'MLP', 'GNN (Proposed)']
accuracy = [78, 86, 89, 87, round(gnn_data['Accuracy']*100, 2)]
precision = [0.73, 0.82, 0.84, 0.83, gnn_data['Precision']]
recall = [0.68, 0.79, 0.83, 0.80, gnn_data['Recall']]
f1 = [0.70, 0.80, 0.83, 0.81, gnn_data['F1']]
auc_scores = [0.79, 0.87, 0.90, 0.88, gnn_data['AUC']]
train_time = [12, 45, 60, 55, gnn_data['Train_Time']]
fp_rate      = [0.22, 0.15, 0.11, 0.13, 1 - gnn_data['Precision']]  # Approx FP rate
fn_rate      = [0.32, 0.21, 0.17, 0.19, 1 - gnn_data['Recall']]     # Approx FN rate
fpr_tpr      = [(1-precision[i], recall[i]) for i in range(len(models))]  # Simple ROC points

sns.set(style="whitegrid")
plt.rcParams.update({'figure.figsize':(8,5)})

# ---------- Accuracy ----------
plt.bar(models, accuracy, color='teal')
plt.ylabel('Accuracy (%)'); plt.title('Model Accuracy Comparison'); plt.xticks(rotation=15)
plt.tight_layout(); 
plt.savefig("comparisongraphs/graph1_auc.png")
plt.clf()  # clears figure for next plot


# ---------- Precision vs Recall ----------
x = np.arange(len(models)); width = 0.35
plt.bar(x - width/2, precision, width, label='Precision')
plt.bar(x + width/2, recall, width, label='Recall')
plt.xticks(x, models, rotation=15); plt.ylabel('Score')
plt.title('Precision vs Recall'); plt.legend(); plt.tight_layout(); 
plt.savefig("comparisongraphs/graph2_prec.png")
plt.clf()  # clears figure for next plot


# ---------- F1 Score ----------
plt.bar(models, f1, color='orange'); plt.ylabel('F1-Score')
plt.title('F1-Score Comparison'); plt.xticks(rotation=15)
plt.tight_layout(); 
plt.savefig("comparisongraphs/graph3_f1score.png")
plt.clf()  # clears figure for next plot


# ---------- AUC ----------
plt.bar(models, auc_scores, color='purple')
plt.ylabel('AUC Score'); plt.title('AUC Comparison'); plt.xticks(rotation=15)
plt.tight_layout(); 
plt.savefig("comparisongraphs/graph4_auc.png")
plt.clf()  # clears figure for next plot


# ---------- Training Time ----------
plt.bar(models, train_time, color='gray')
plt.ylabel('Training Time (s)'); plt.title('Model Training Time'); plt.xticks(rotation=15)
plt.tight_layout(); 
plt.savefig("comparisongraphs/graph5_traintime.png")
plt.clf()  # clears figure for next plot

# ---------- 6. False Positive Rate ----------
plt.bar(models, fp_rate, color='red')
plt.ylabel('False Positive Rate')
plt.title('False Positive Rate Comparison')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("comparisongraphs/graph6_fp_rate.png")
plt.clf()

# ---------- 7. False Negative Rate ----------
plt.bar(models, fn_rate, color='brown')
plt.ylabel('False Negative Rate')
plt.title('False Negative Rate Comparison')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("comparisongraphs/graph7_fn_rate.png")
plt.clf()

# ---------- 8. Precision / F1 Ratio ----------
ratio = [precision[i]/f1[i] for i in range(len(models))]
plt.bar(models, ratio, color='pink')
plt.ylabel('Precision / F1 Ratio')
plt.title('Precision vs F1 Ratio')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("comparisongraphs/graph8_precision_f1_ratio.png")
plt.clf()

# ---------- 9. Recall / F1 Ratio ----------
ratio2 = [recall[i]/f1[i] for i in range(len(models))]
plt.bar(models, ratio2, color='lightgreen')
plt.ylabel('Recall / F1 Ratio')
plt.title('Recall vs F1 Ratio')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("comparisongraphs/graph9_recall_f1_ratio.png")
plt.clf()

# ---------- 10. Simple ROC Points ----------
plt.figure()
for i, m in enumerate(models):
    plt.scatter(fpr_tpr[i][0], fpr_tpr[i][1], label=m)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Points Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("comparisongraphs/graph10_roc_points.png")
plt.clf()

print("✅ All 10 graphs saved in 'comparisongraphs/' folder.")