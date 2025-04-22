import os
import cv2
import torch
import numpy as np
import tensorflow as tf
import albumentations as A
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset
from tensorflow import keras
from tqdm import tqdm


# Проведение исследований с моделями семантической сегментации из segmentation_models.pytorch"

class FireDataset(Dataset):
    """Кастомный датасет для загрузки изображений и масок пожаров"""

    def __init__(self, ids, image_path, mask_path, image_size=128):
        """Инициализация датасета
        
        Args:
            ids (list): Список идентификаторов изображений
            image_path (str): Путь к директории с изображениями
            mask_path (str): Путь к директории с масками
            image_size (int): Размер, к которому будут приводиться изображения
        """
        self.ids = ids
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_size = image_size

    def __len__(self):
        """Вычисление количества элементов в датасете"""
        return len(self.ids)

    def __getitem__(self, idx):
        """Загрузка одного изображения и соответствующей маски
        
        Args:
            idx (int): Индекс элемента
            
        Returns:
            tuple: (image, mask) - тензоры изображения и маски
        """
        id_name = self.ids[idx]
        image = cv2.imread(os.path.join(self.image_path, id_name), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW

        mask = cv2.imread(os.path.join(self.mask_path, id_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = (mask > 0).astype(np.float32)  # Бинаризация
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # Добавляем размерность канала

        return image, mask

def iou_torch(y_true, y_pred):
    """Вычисление метрики IoU (Intersection over Union) для torch"""
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return intersection / (union + 1e-6)

def dice_torch(y_true, y_pred):
    """Вычисление метрики Dice коэффициент для torch"""
    intersection = torch.sum(y_true * y_pred)
    return 2 * intersection / (torch.sum(y_true) + torch.sum(y_pred) + 1e-6)

def precision_torch(y_true, y_pred):
    """Вычисление метрики Precision (точность) для torch"""
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    return precision_score(y_true.cpu(), y_pred.cpu(), average='binary')

def recall_torch(y_true, y_pred):
    """Вычисление метрики Recall (полнота) для torch"""
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    return recall_score(y_true.cpu(), y_pred.cpu(), average='binary')

def f1_torch(y_true, y_pred):
    """Вычисление метрики F1-score (гармоническое среднее precision и recall) для torch"""
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    return f1_score(y_true.cpu(), y_pred.cpu(), average='binary')

def train_model_torch(model, train_loader, val_loader, optimizer, loss_fn, epochs=10, device='cuda'):
    """Функция для обучения модели с валидацией
    
    Args:
        model: Модель для обучения
        train_loader: DataLoader для обучающих данных
        val_loader: DataLoader для валидационных данных
        optimizer: Оптимизатор
        loss_fn: Функция потерь
        epochs (int): Количество эпох обучения
        device: Устройство, на котором будут проводиться вычисления
        
    Returns:
        None: Выводит прогресс обучения и метрики в консоль
    """
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

                pred = torch.sigmoid(outputs) > 0.5
                val_metrics['iou'] += iou_torch(masks, pred).item()
                val_metrics['dice'] += dice_torch(masks, pred).item()
                val_metrics['precision'] += precision_torch(masks, pred)
                val_metrics['recall'] += recall_torch(masks, pred)
                val_metrics['f1'] += f1_torch(masks, pred)  # Добавлено вычисление F1

            num_val_batches = len(val_loader)
            val_loss /= num_val_batches
            for key in val_metrics:
                val_metrics[key] /= num_val_batches

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")


# Собственная имплементация моделей семантической сегментации

class DataGen(keras.utils.Sequence):
    """Генератор данных для загрузки и обработки изображений и масок в реальном времени"""
 
    def __init__(self, ids, path, batch_size=8, image_size=128):
        """Инициализация генератора данных
        
        Args:
            ids (list): Список идентификаторов изображений
            path (str): Базовый путь к директории с данными
            batch_size (int): Размер батча
            image_size (int): Размер, к которому будут приводиться изображения
        """
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __load__(self, id_name):
        """Загрузка и обработка одного изображения и соответствующей маски
        
        Args:
            id_name (str): Идентификатор изображения
            
        Returns:
            tuple: (image, mask) - обработанные изображение и маска
        """
        image_path = os.path.join("/content/fire-segmentation-dataset/fire-segmentation-image-dataset/Image/Fire", id_name)
        mask_path = os.path.join("/content/fire-segmentation-dataset/fire-segmentation-image-dataset/Segmentation_Mask/Fire", id_name)
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask = np.zeros((self.image_size, self.image_size, 1))

        _mask_image = cv2.imread(mask_path, -1)
        _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size))
        _mask_image = np.expand_dims(_mask_image, axis=-1)
        mask = np.maximum(mask, _mask_image)


        image = image/255.0
        mask = mask/255.0

        return image, mask

    def __getitem__(self, index):
        """Возврат одного батча данных
        
        Args:
            index (int): Индекс батча
            
        Returns:
            tuple: (images, masks) - массивы изображений и масок батча
        """
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size

        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]

        image = []
        mask  = []

        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask  = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        """Вызывается в конце каждой эпохи (может использоваться для перемешивания данных)"""
        pass

    def __len__(self):
        """Вычисление количества батчей в эпохе"""
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
    
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """Нисходящий блок U-Net архитектуры (кодировщик)
    
    Args:
        x: Входной тензор
        filters: Количество фильтров
        kernel_size: Размер ядра свертки
        padding: Тип паддинга
        strides: Шаг свертки
        
    Returns:
        tuple: (c, p) - результат сверток и результат макс-пулинга
    """
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    """Восходящий блок U-Net архитектуры (декодировщик) с skip-связями
    
    Args:
        x: Входной тензор
        skip: Тензор из соответствующего нисходящего блока
        filters: Количество фильтров
        kernel_size: Размер ядра свертки
        padding: Тип паддинга
        strides: Шаг свертки
        
    Returns:
        tensor: Результат апсемплинга и сверток
    """
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """Бутылочное горлышко U-Net архитектуры (самый нижний слой)
    
    Args:
        x: Входной тензор
        filters: Количество фильтров
        kernel_size: Размер ядра свертки
        padding: Тип паддинга
        strides: Шаг свертки
        
    Returns:
        tensor: Результат двух сверток
    """
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def iou(y_true, y_pred):
    """ Вычисление метрики IoU (Intersection over Union) для keras """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def dice_coef(y_true, y_pred):
    """ Вычисление метрики Dice коэффициент для keras """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)

def precision(y_true, y_pred):
    """ Вычисление метрики Precision для keras """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    return (true_positives + 1e-7) / (predicted_positives + 1e-7)

def recall(y_true, y_pred):
    """ Вычисление метрики Recall для keras """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    actual_positives = tf.reduce_sum(y_true)
    return (true_positives + 1e-7) / (actual_positives + 1e-7)

def custom_cnn_segmentation_model(image_size=128):
    """Создание кастомной U-Net модели для сегментации

    Args:
        image_size (int): Размер изображения
    
    Returns:
        model: Готовая Keras модель
    """
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128 --> 64
    c2, p2 = down_block(p1, f[1])  # 64  --> 32
    c3, p3 = down_block(p2, f[2])  # 32  --> 16
    c4, p4 = down_block(p3, f[3])  # 16  --> 8

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])  # 8  --> 16
    u2 = up_block(u1, c3, f[2])  # 16 --> 32
    u3 = up_block(u2, c2, f[1])  # 32 --> 64
    u4 = up_block(u3, c1, f[0])  # 64 --> 128

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

def custom_train_loop(model, train_gen, valid_gen, epochs, train_steps, valid_steps):
    """Кастомный цикл обучения модели с отслеживанием метрик
    
    Args:
        model: Модель для обучения
        train_gen: Генератор обучающих данных
        valid_gen: Генератор валидационных данных
        epochs: Количество эпох
        train_steps: Количество шагов обучения на эпоху
        valid_steps: Количество шагов валидации на эпоху
    """
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Метрики
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_iou = tf.keras.metrics.MeanIoU(num_classes=2, name='train_iou')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_iou = tf.keras.metrics.MeanIoU(num_classes=2, name='val_iou')
    val_precision = tf.keras.metrics.Precision(name='val_precision')
    val_recall = tf.keras.metrics.Recall(name='val_recall')
    val_dice = tf.keras.metrics.Mean(name='val_dice')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Обучение
        train_loss.reset_state()
        train_iou.reset_state()
        for step in tqdm(range(train_steps), desc="Training"):
            x_batch, y_batch = train_gen.__getitem__(step)

            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss = loss_fn(y_batch, preds)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Обновление метрик
            train_loss.update_state(loss)
            train_iou.update_state(y_batch, preds > 0.5)

        # Валидация
        val_loss.reset_state()
        val_iou.reset_state()
        val_precision.reset_state()
        val_recall.reset_state()
        val_dice.reset_state()
        for step in tqdm(range(valid_steps), desc="Validation"):
            x_val, y_val = valid_gen.__getitem__(step)
            val_preds = model(x_val, training=False)

            # Обновление метрик валидации
            val_loss.update_state(loss_fn(y_val, val_preds))
            val_iou.update_state(y_val, val_preds > 0.5)
            val_precision.update_state(y_val, val_preds > 0.5)
            val_recall.update_state(y_val, val_preds > 0.5)
            val_dice.update_state(dice_coef(y_val, val_preds))

        # Вывод метрик
        print(f"Train Loss: {train_loss.result():.4f} | Train IoU: {train_iou.result():.4f}")
        print(f"Val Loss: {val_loss.result():.4f} | Val IoU: {val_iou.result():.4f} | "
              f"Val Precision: {val_precision.result():.4f} | Val Recall: {val_recall.result():.4f} | "
              f"Val Dice: {val_dice.result():.4f}")


class TransformerBlock(keras.layers.Layer):
    """Блок трансформера с self-attention и feed-forward сетями"""

    def __init__(self, num_heads, key_dim, ff_dim):
        """Инициализация блока трансформера
        
        Args:
            num_heads (int): Количество голов внимания
            key_dim (int): Размерность ключей/запросов
            ff_dim (int): Размерность скрытого слоя FFN
        """
        super(TransformerBlock, self).__init__()
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = keras.models.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(key_dim)
        ])
        self.layernorm1 = keras.layers.LayerNormalization()
        self.layernorm2 = keras.layers.LayerNormalization()
        self.dropout1 = keras.layers.Dropout(0.1)
        self.dropout2 = keras.layers.Dropout(0.1)

    def call(self, inputs, training=False):
        """Прямой проход блока трансформера
        
        Args:
            inputs: Входной тензор
            training (bool): Флаг режима обучения
            
        Returns:
            Тензор после обработки блоком трансформера
        """
        # Внимание
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Skip connection

        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Skip connection
    
def custom_transformer_segmentation_model(image_size, num_heads=4, key_dim=64, ff_dim=256):
    """Создание кастомной модели сегментации с трансформерным блоком
    
    Args:
        image_size (int): Размер входного изображения
        num_heads (int): Количество голов внимания
        key_dim (int): Размерность ключей/запросов
        ff_dim (int): Размерность скрытого слоя FFN
        
    Returns:
        model: Готовая Keras модель
    """
    inputs = keras.layers.Input(shape=(image_size, image_size, 3))

    # Сверточные блоки (первоначальная обработка)
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Преобразуем для использования трансформера
    x = keras.layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)  # [batch_size, height*width, channels]

    # Блок трансформера
    x = TransformerBlock(num_heads, key_dim, ff_dim)(x)

    # Возвращаем в размерность изображения
    x = keras.layers.Reshape((image_size // 2, image_size // 2, 64))(x)

    # Декодеры и сегментация
    x = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)

    model = keras.models.Model(inputs, x)
    return model


# Улучшение бейзлайна

class AugmentedFireDataGen(keras.utils.Sequence):
    """Генератор данных с аугментациями для сегментации пожаров"""

    def __init__(self, ids, path, batch_size=16, image_size=64, augment=True):
        """Инициализация генератора данных
        
        Args:
            ids (list): Список идентификаторов изображений
            path (str): Базовый путь к директории с данными
            batch_size (int): Размер батча
            image_size (int): Размер изображений
            augment (bool): Флаг применения аугментаций
        """
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.on_epoch_end()

        # Специализированные аугментации для огня
        self.transform = A.Compose([
            # Геометрические преобразования
            A.HorizontalFlip(p=0.4),
            A.Rotate(limit=15, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=0.5),

            # Яркость/контрастность
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.3
            ),

            # Шум и эффекты
            A.GaussNoise(var_limit=(0.001, 0.005), p=0.2),
            A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.1, p=0.1),

            # Деформации пламени
            A.ElasticTransform(
                alpha=15,
                sigma=5,
                alpha_affine=3,
                interpolation=cv2.INTER_NEAREST,
                p=0.2
            )
        ], additional_targets={'mask': 'mask'})

    def __load__(self, id_name):
        """Загрузка и обработка одного изображения и соответствующей маски
        
        Args:
            id_name (str): Идентификатор изображения
            
        Returns:
            tuple: (image, mask) - обработанные изображение и маска
        """
        image_path = os.path.join(self.path, "Image/Fire", id_name)
        mask_path = os.path.join(self.path, "Segmentation_Mask/Fire", id_name)

        # Загрузка изображения
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Загрузка маски
        mask = np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)
        _mask_image = cv2.imread(mask_path, -1)
        _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        _mask_image = np.expand_dims(_mask_image, axis=-1)
        mask = np.maximum(mask, _mask_image)

        # Применение аугментаций
        if self.augment:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Нормализация
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        return image, mask

    def __getitem__(self, index):
        """Возврат одного батча данных
        
        Args:
            index (int): Индекс батча
            
        Returns:
            tuple: (images, masks) - массивы изображений и масок
        """
        if (index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size

        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]

        images = []
        masks = []

        for id_name in files_batch:
            img, mask = self.__load__(id_name)
            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        """Действия в конце каждой эпохи (перемешивание данных)"""
        if self.augment:
            np.random.shuffle(self.ids)

    def __len__(self):
        """Вычисление количества батчей в эпохе"""
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

class TransformerBlockWithSkip(keras.layers.Layer):
    """Улучшенный блок трансформера с skip-связями и настройкой dropout"""

    def __init__(self, num_heads, key_dim, ff_dim, dropout_rate=0.1):
        """Инициализация блока трансформера
        
        Args:
            num_heads (int): Количество голов внимания
            key_dim (int): Размерность ключей/запросов
            ff_dim (int): Размерность скрытого слоя FFN
            dropout_rate (float): Вероятность dropout
        """
        super(TransformerBlockWithSkip, self).__init__()
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = keras.models.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(key_dim)
        ])
        self.layernorm1 = keras.layers.LayerNormalization()
        self.layernorm2 = keras.layers.LayerNormalization()
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        """Прямой проход блока трансформера
        
        Args:
            inputs: Входной тензор
            training (bool): Флаг режима обучения
            
        Returns:
            Тензор после обработки блоком трансформера
        """
        # Self-attention with skip connection
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # FFN with skip connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def improved_custom_transformer(image_size=128):
    """Создание улучшенной модели сегментации с трансформерами
    
    Args:
        image_size (int): Размер входного изображения
        
    Returns:
        model: Готовая Keras модель
    """
    inputs = keras.layers.Input((image_size, image_size, 3))

    ### Энкодер (глубокая CNN с регуляризацией) ###
    # Блок 1
    x1 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same',
                           kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
    x1 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x1)
    x1 = keras.layers.BatchNormalization()(x1)
    p1 = keras.layers.MaxPooling2D((2,2))(x1)
    p1 = keras.layers.Dropout(0.1)(p1)

    # Блок 2
    x2 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same',
                           kernel_regularizer=keras.regularizers.l2(1e-4))(p1)
    x2 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    p2 = keras.layers.MaxPooling2D((2,2))(x2)
    p2 = keras.layers.Dropout(0.2)(p2)

    ### Трансформерный блок (2 слоя) ###
    _, h, w, c = p2.shape
    x = keras.layers.Reshape((h*w, c))(p2)
    x = TransformerBlockWithSkip(num_heads=4, key_dim=128, ff_dim=512)(x)  # Увеличенные размерности
    x = TransformerBlockWithSkip(num_heads=4, key_dim=128, ff_dim=512)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Reshape((h, w, c))(x)

    ### Декодер с skip-connections ###
    # Up Блок 1
    u1 = keras.layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(x)
    u1 = keras.layers.Concatenate()([u1, x2])  # Skip-connection
    u1 = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(u1)
    u1 = keras.layers.BatchNormalization()(u1)
    u1 = keras.layers.Dropout(0.2)(u1)

    # Up Блок 2
    u2 = keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(u1)
    u2 = keras.layers.Concatenate()([u2, x1])  # Skip-connection
    u2 = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(u2)
    u2 = keras.layers.BatchNormalization()(u2)

    ### Выход ###
    outputs = keras.layers.Conv2D(1, (1,1), activation='sigmoid')(u2)

    model = keras.models.Model(inputs, outputs)
    return model
