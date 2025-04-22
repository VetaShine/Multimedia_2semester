import torch
import time
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


def calculate_metrics(y_true, y_pred, num_classes=None):
    """Функция подсчета метрик обучения"""
    if num_classes is None:
        num_classes = max(max(y_true), max(y_pred)) + 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion_mat[t, p] += 1

    # Accuracy
    accuracy = np.trace(confusion_mat) / np.sum(confusion_mat)

    # Precision, Recall, F1 (взвешенные)
    precision_list = []
    recall_list = []
    f1_list = []
    weights = []

    for i in range(num_classes):
        tp = confusion_mat[i, i]
        fp = confusion_mat[:, i].sum() - tp
        fn = confusion_mat[i, :].sum() - tp
        support = confusion_mat[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        weights.append(support)

    weights = np.array(weights)
    total = np.sum(weights)
    weighted_precision = np.sum(np.array(precision_list) * weights) / total
    weighted_recall = np.sum(np.array(recall_list) * weights) / total
    weighted_f1 = np.sum(np.array(f1_list) * weights) / total

    return accuracy, weighted_precision, weighted_recall, weighted_f1, confusion_mat

def plot_confusion_matrix(confusion_mat, classes):
    """Функция для визуализации матрицы ошибок"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def evaluate_model(model, test_loader, criterion, device='cuda'):
    """Функция для оценки модели на тестовом наборе"""
    model.eval()
    y_true, y_pred = [], []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy, test_precision, test_recall, test_f1, test_confusion = calculate_metrics(y_true, y_pred)

    return test_loss, test_accuracy, test_precision, test_recall, test_f1, test_confusion

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, test_dataset, device='cuda', num_epochs=10):
    """Функция для обучения модели"""
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        y_true, y_pred = [], []

        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Backward pass и обновление градиентов
            loss.backward()
            optimizer.step()

        # Подсчет метрик на обучении
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy, train_precision, train_recall, train_f1, train_confusion = calculate_metrics(y_true, y_pred)

        # Оценка на тестовой выборке
        model.eval()
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_confusion = evaluate_model(model, test_loader, criterion)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, "
              f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}, "
              f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

        # Визуализация матрицы ошибок для тестовой выборки
        plot_confusion_matrix(test_confusion, test_dataset.classes)

        print(f"Time taken for epoch: {time.time() - start_time:.2f} sec\n")

def train_improved_cnn_model(model, train_loader, test_loader, criterion, optimizer, scheduler, test_dataset, device='cuda', num_epochs=10):
    """Улучшенная версия функции обучения для класса ImprovedCNN"""
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        running_loss = 0.0
        y_true, y_pred = [], []

        start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Backward pass и обновление градиентов
            loss.backward()
            optimizer.step()

        # Подсчет метрик на обучении
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy, train_precision, train_recall, train_f1, train_confusion = calculate_metrics(y_true, y_pred)

        # Оценка на тестовой выборке
        model.eval()
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_confusion = evaluate_model(model, test_loader, criterion)

        # Обновление learning rate
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(test_accuracy)
        else:
            scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Train F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
              f"Test F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        print(f"Time taken: {time.time() - start_time:.2f} sec\n")

        # Визуализация матрицы ошибок
        plot_confusion_matrix(test_confusion, test_dataset.classes)

    return model

class CustomCNN(nn.Module):
    """Кастомная сверточная нейросеть (CNN) для классификации изображений на 15 классов. Состоит из трех сверточных блоков с BatchNorm, ReLU и Dropout, а также полносвязной классификаторной головы."""

    def __init__(self, num_classes=15):
        """ Инициализация кастомной CNN-модели"""
        super(CustomCNN, self).__init__()

        # Улучшенная сверточная часть
        self.features = nn.Sequential(
            # Блок 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # Блок 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # Блок 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

        # Полносвязная часть
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Прямой проход через сеть: экстракция признаков -> выпрямление -> классификация"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class PatchEmbedding(nn.Module):
    """Класс для разбиения изображения на патчи и приведения их к эмбеддингам фиксированной размерности. Используется в Vision Transformer для замены сверточных слоев."""

    def __init__(self, in_channels=3, patch_size=8, emb_size=768, img_size=64):
        """Инициализация слоя патч-эмбеддинга для Vision Transformer"""
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, emb_size))

    def forward(self, x):
        """Преобразует входное изображение в последовательность эмбеддингов патчей и добавляет специальный [CLS]-токен и позиционные эмбеддинги"""
        B = x.shape[0]
        x = self.proj(x)  # (B, emb_size, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, emb_size)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + N_patches, emb_size)
        x += self.pos_embedding
        return x

class TransformerEncoderBlock(nn.Module):
    """Один блок энкодера трансформера, содержащий слой нормализации, многоголовую самовнимательность и MLP"""

    def __init__(self, emb_size=768, num_heads=8, dropout=0.1, forward_expansion=4):
        """Инициализация одного блока энкодера трансформера"""
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * forward_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * forward_expansion, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Прямой проход через блок: нормализация -> внимание -> нормализация -> MLP"""
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Кастомная реализация Vision Transformer (ViT) для классификации изображений. Использует патч-эмбеддинги, энкодеры и классификационную голову"""

    def __init__(self,
                 img_size=64,
                 patch_size=16,
                 in_channels=3,
                 num_classes=15,
                 emb_size=768,
                 depth=6,
                 num_heads=8,
                 dropout=0.1):
        """Инициализация кастомного Vision Transformer"""
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_size=emb_size,
            img_size=img_size
        )

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(emb_size, num_heads, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        """Прямой проход через модель: патч-эмбеддинги -> энкодеры -> нормализация -> классификация"""
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

class ImprovedCNN(nn.Module):
    """Улучшенная версия кастомной CNN с более глубокой архитектурой, большим количеством фильтров и использованием LeakyReLU. Предназначена для повышения точности классификации по сравнению с базовой моделью."""

    def __init__(self, num_classes=15):
        """Инициализация улучшенной версии кастомной CNN"""
        super(ImprovedCNN, self).__init__()

        # Улучшенная сверточная часть с более глубокой архитектурой
        self.features = nn.Sequential(
            # Блок 1 с увеличенным количеством фильтров
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            # Блок 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            # Блок 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),

            # Блок 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
        )

        # Улучшенная полносвязная часть
        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Прямой проход: извлечение признаков через сверточную часть -> выпрямление -> классификация"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
