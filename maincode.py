import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备

# 模型参数
num_classes = 5
max_length = 128
batch_size = 16
hidden_dim = 768  # 与BioBERT输出维度一致


# 标签映射
label_map = {
    'DDI-false': 0,
    'DDI-effect': 1,
    'DDI-mechanism': 2,
    'DDI-advise': 3,
    'DDI-int': 4
}

# BioBERT模型路径
model_path = './model_path/biobert'
biobert_tokenizer = AutoTokenizer.from_pretrained(model_path)
biobert_model = AutoModel.from_pretrained(model_path).to(device)


#读取数据
df_train = pd.read_csv('./data/ddi2013ms/train.tsv', sep='\t')
df_dev = pd.read_csv('./data/ddi2013ms/dev.tsv', sep='\t')
df_test = pd.read_csv('./data/ddi2013ms/test.tsv', sep='\t')
print("read")

# print("训练集数据量：", df_train.shape)
# print("验证集数据量：", df_dev.shape)
# print("测试集数据量：", df_test.shape)


# 数据加载器
# 定义数据集类（修正版本）
class DDIDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['sentence']
        label = label_map[self.data.iloc[idx]['label']]
        encoding = self.tokenizer(sentence,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# DataLoader函数
def create_data_loader(df, tokenizer, max_length, batch_size):
    dataset = DDIDataset(df=df, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

train_data_loader = create_data_loader(df_train, biobert_tokenizer, 128, 32)
dev_data_loader = create_data_loader(df_dev, biobert_tokenizer, 128, 32)
test_data_loader = create_data_loader(df_test, biobert_tokenizer, 128, 32)


# 模型定义：BioBERT + ProtoNet + Contrastive Learning
class ProtoContrastiveDDI(nn.Module):
    def __init__(self, encoder, hidden_dim=768, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)  # 分类器头部

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.projection(outputs.pooler_output)  # 可用于对比学习
        return pooled  # 返回嵌入用于后续分类 or 损失



# Prototypical Network的原型计算函数
def compute_prototypes(support_embeddings, support_labels):
    prototypes = []
    for label in torch.unique(support_labels):
        prototypes.append(support_embeddings[support_labels == label].mean(0))
    prototypes = torch.stack(prototypes)
    return prototypes

# Prototypical Loss函数
def proto_loss(query_embeddings, prototypes, query_labels):
    unique_labels = torch.unique(query_labels)
    label_mapping = {old.item(): new for new, old in enumerate(unique_labels)}
    mapped_labels = torch.tensor([label_mapping[label.item()] for label in query_labels], device=query_labels.device)
    distances = torch.cdist(query_embeddings, prototypes)
    return F.cross_entropy(-distances, mapped_labels)


# InfoNCE 对比损失函数
def contrastive_loss(features, labels, temperature=0.07):
    features_norm = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
    loss = F.cross_entropy(similarity_matrix, labels_matrix.float())
    return loss

# 模型实例化
model = ProtoContrastiveDDI(biobert_model).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)


# 训练代码
def train_model(model, train_data_loader, optimizer, criterion, device, proto_weight=0.5, contrastive_weight=0.1):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    epoch_true_labels = []
    epoch_pred_labels = []
    epoch_true_probs = []

    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # 获取嵌入
        embeddings = model(input_ids, attention_mask)  # [batch_size, hidden_dim]

        # 分类器头部
        logits = model.classifier(embeddings)

        # 分类损失
        ce_loss = criterion(logits, labels)

        # 原型损失
        prototypes = compute_prototypes(embeddings, labels)
        proto_loss_val = proto_loss(embeddings, prototypes, labels)

        # 对比损失
        contrast_loss_val = contrastive_loss(embeddings, labels)

        # 总损失 = 分类 + 原型 + 对比
        total_loss = ce_loss + proto_weight * proto_loss_val + contrastive_weight * contrast_loss_val
        total_loss.backward()
        optimizer.step()

        # 指标统计
        running_loss += total_loss.item()
        _, predicted = torch.max(logits, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        epoch_true_labels.extend(labels.cpu().numpy())
        epoch_pred_labels.extend(predicted.cpu().numpy())
        epoch_true_probs.extend(F.softmax(logits, dim=1).detach().cpu().numpy())

    train_loss = running_loss / len(train_data_loader)
    train_acc = correct_preds / total_preds

    train_conf_matrix = confusion_matrix(epoch_true_labels, epoch_pred_labels)
    train_precision = precision_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    train_recall = recall_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    train_f1 = f1_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)

    return train_loss, train_acc, train_conf_matrix, train_precision, train_recall, train_f1






# 测试代码
def test_model(model, test_data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    epoch_true_labels = []
    epoch_pred_labels = []
    epoch_predicted_probs = []

    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 获取嵌入（特征表示）
            embeddings = model(input_ids, attention_mask)

            # 分类器推理
            logits = model.classifier(embeddings)

            # 损失计算
            loss = criterion(logits, labels)
            running_loss += loss.item()

            # 预测标签
            _, predicted = torch.max(logits, dim=1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            # 收集数据用于评估
            epoch_true_labels.extend(labels.cpu().numpy())
            epoch_pred_labels.extend(predicted.cpu().numpy())
            epoch_predicted_probs.extend(F.softmax(logits, dim=1).detach().cpu().numpy())

    test_loss = running_loss / len(test_data_loader)
    test_accuracy = accuracy_score(epoch_true_labels, epoch_pred_labels)
    test_precision = precision_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    test_recall = recall_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    test_f1 = f1_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    test_conf_matrix = confusion_matrix(epoch_true_labels, epoch_pred_labels)

    # 每类F1
    class_report = classification_report(epoch_true_labels, epoch_pred_labels, output_dict=True, zero_division=1)
    test_f1_per_class = {label: metrics['f1-score'] for label, metrics in class_report.items() if label.isdigit()}

    return test_conf_matrix, test_accuracy, test_precision, test_recall, test_f1, test_f1_per_class, epoch_true_labels, epoch_predicted_probs




### 完整的训练循环示例
num_epochs = 30
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()


# 存储训练过程中每个 epoch 的结果用于画图
epoch_train_losses = []
epoch_train_accuracies = []
epoch_train_f1_scores = []

# 存储测试集真实标签与预测概率 (用于画ROC曲线)
true_labels = []
predicted_probs = []

# 打开训练结果文件，以追加模式写入
with open('training_results.txt', 'a') as f_train:
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):

        train_loss, train_acc, train_conf_matrix, train_precision, train_recall, train_f1 = train_model(
            model, train_data_loader, optimizer, criterion, device
        )

        # 保存训练结果
        f_train.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}\n")
        f_train.write("Confusion Matrix:\n")
        f_train.write(str(train_conf_matrix) + '\n')

        # 打印结果
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {train_f1:.4f}")
        print("Confusion Matrix:")
        print(train_conf_matrix)
        
        # 收集用于绘图的数据
        epoch_train_losses.append(train_loss)
        epoch_train_accuracies.append(train_acc)
        epoch_train_f1_scores.append(train_f1)

# 单独保存测试结果
with open('test_results.txt', 'w') as f_test:
    test_conf_matrix, test_accuracy, test_precision, test_recall, test_f1, test_f1_per_class, epoch_true_labels, epoch_predicted_probs = test_model(model, test_data_loader, criterion, device)

    true_labels = epoch_true_labels
    predicted_probs = epoch_predicted_probs

    f_test.write("Test Results:\n")
    f_test.write("Confusion Matrix:\n")
    f_test.write(str(test_conf_matrix) + '\n')
    f_test.write(f"Accuracy: {test_accuracy:.4f}\n")
    f_test.write(f"Precision: {test_precision:.4f}\n")
    f_test.write(f"Recall: {test_recall:.4f}\n")
    f_test.write(f"F1 Score: {test_f1:.4f}\n")
    f_test.write("F1 Score per Class:\n")
    for label, f1_cls in test_f1_per_class.items():
        f_test.write(f"Class {label}: {f1_cls:.4f}\n")


    # 控制台打印测试结果
    print("Test Results:")
    print("Confusion Matrix:")
    print(test_conf_matrix)
    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1 Score:", test_f1)
    print("F1 Score per Class:")
    for label, f1_cls in test_f1_per_class.items():
        print(f"Class {label}: {f1_cls:.4f}")

#########################################训练集画图##################################################
# 统一色调
colors = sns.color_palette("Set2", n_colors=5)
class_names = list(label_map.keys())


# 训练Loss曲线图
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_losses, marker='o', label='Train Loss', color=colors[0])
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training Loss Over Epochs', fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=300)
plt.show()

# 训练准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_accuracies, marker='o', label='Train Accuracy', color=colors[1])
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Training Accuracy Over Epochs', fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('training_accuracy.png', dpi=300)
plt.show()

# 训练F1-score曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_f1_scores, marker='o', label='Train F1 Score', color=colors[2])
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.title('Training F1 Score Over Epochs', fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('training_f1_score.png', dpi=300)
plt.show()

########################################## 测试集画图 ######################################################

# 混淆矩阵热力图
plt.figure(figsize=(10, 8))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix (Test Set)', fontsize=16)
plt.tight_layout()
plt.savefig('test_confusion_matrix.png', dpi=300)
plt.show()


# 准确率柱状图（测试集整体准确率）
plt.figure(figsize=(8, 6))
plt.bar(class_names, [test_accuracy]*len(class_names), color=colors)
plt.xlabel('Class', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Test Accuracy by Class', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('test_accuracy.png', dpi=300)
plt.show()


# 测试集ROC曲线图
fpr, tpr, roc_auc = dict(), dict(), dict()

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    class_true_labels = [1 if lbl == i else 0 for lbl in true_labels]
    class_probs = [prob[i] for prob in predicted_probs]

    fpr[i], tpr[i], _ = roc_curve(class_true_labels, class_probs)
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC={roc_auc[i]:.2f})', color=colors[i])

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curves by Class (Test Set)', fontsize=16)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('test_roc_curve.png', dpi=300)
plt.show()



