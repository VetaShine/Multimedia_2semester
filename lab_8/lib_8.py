import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random
import torch
import torch.nn as nn
import torchmetrics
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet50


def display_sample_images(directory, num_images=5):
    """Function to display a few images from the specified directory"""
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_files = image_files[:num_images]  # Select a few images

    # Display the images using matplotlib
    plt.figure(figsize=(10, 5))
    for idx, image_file in enumerate(selected_files):
        img = Image.open(os.path.join(directory, image_file))
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {idx + 1}")
    plt.show()

def create_dataset(root_dir, dataset_name, split, num_images, background_dir, objects, classes):
    """Create a directory for saving synthetic images"""
    synthetic_dir = os.path.join(root_dir, dataset_name, split)
    image_dir = os.path.join(synthetic_dir, 'images')
    label_dir = os.path.join(synthetic_dir, 'labels')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    def add_object_to_background(background, object_img):
        """Adding the objects to the background"""
        if object_img.width > background.width or object_img.height > background.height:
            return None  # Skip this background

        # Get random position for placing the object on the background
        max_x = background.width - object_img.width
        max_y = background.height - object_img.height

        # If max_x or max_y are negative, skip placing the object
        if max_x < 0 or max_y < 0:
            return None  # Skip this background

        position = (random.randint(0, max_x), random.randint(0, max_y))

        # Paste the object onto the background using transparency (alpha channel)
        background.paste(object_img, position, object_img)
        return background, position, object_img.size

    for i in range(num_images):
        # Load a random background image
        background_file = random.choice([f for f in os.listdir(background_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        background_path = os.path.join(background_dir, background_file)
        background = Image.open(background_path).convert("RGBA")

        # Select a random object
        random_object_name = random.choice(classes)
        object_path = objects[random_object_name]
        object_img = Image.open(object_path).convert("RGBA")  # Ensure alpha channel (transparency)

        # Add the selected object to the background
        result = add_object_to_background(background, object_img)

        if result is None:
            print(f"Skipping background {background_file} as it's too small for the object.")
            continue  # Skip this background and continue with the next one

        synthetic_image, (x, y), (w, h) = result
        synthetic_image = synthetic_image.convert("RGB")  # Convert to RGB before saving

        # Save the image
        image_filename = f"{i:05d}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        synthetic_image.save(image_path)

        # Calculate YOLO-format bounding box using the original background dimensions
        img_width, img_height = synthetic_image.width, synthetic_image.height
        yolo_x = (x + w / 2) / img_width
        yolo_y = (y + h / 2) / img_height
        yolo_w = w / img_width
        yolo_h = h / img_height
        class_index = classes.index(random_object_name)

        # Save the label in YOLO format
        label_path = os.path.join(label_dir, f"{i:05d}.txt")
        with open(label_path, "w") as f:
            f.write(f"{class_index} {yolo_x} {yolo_y} {yolo_w} {yolo_h}\n")

    print(f"Synthetic images and labels saved in: {synthetic_dir}")

def visualize_dataset(dataset, classes, num_images=5):
    """Visualizes a few dataset samples with bounding boxes and class labels drawn on the images"""
    for i in range(num_images):
        image, category, bbox = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        plt.imshow(image)

        # Convert bbox from YOLO format back to coordinates for visualization
        img_height, img_width = image.shape[:2]
        x_center = bbox[0] * img_width
        y_center = bbox[1] * img_height
        box_width = bbox[2] * img_width
        box_height = bbox[3] * img_height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2

        plt.gca().add_patch(
            plt.Rectangle((x1, y1), box_width, box_height, edgecolor='red', facecolor='none', linewidth=2)
        )
        plt.title(f"Class: {classes[category]}")
        plt.show()

def train_model(model, train_loader, val_loader, optimizer, class_loss_fn, bbox_loss_fn, device, num_epochs=30, patience=5, alpha=0.7):
    """Trains a model for object detection with early stopping and logs training/validation losses"""
    train_class_losses = []
    train_bbox_losses = []
    val_class_losses = []
    val_bbox_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    model.train()

    for epoch in range(num_epochs):
        train_class_loss = 0.0
        train_bbox_loss = 0.0
        val_class_loss = 0.0
        val_bbox_loss = 0.0
        model.train()

        # Training loop
        for images, class_labels, bbox_labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            class_labels = class_labels.to(device)
            bbox_labels = bbox_labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            class_preds, bbox_preds = model(images)

            # Calculate losses
            classification_loss = class_loss_fn(class_preds, class_labels)
            bbox_regression_loss = bbox_loss_fn(bbox_preds, bbox_labels)

            # Apply weight to the bounding box loss
            total_loss = alpha * classification_loss + (1 - alpha) * bbox_regression_loss * 100

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            train_class_loss += classification_loss.item() * images.size(0)  # Accumulate class loss for the batch
            train_bbox_loss += bbox_regression_loss.item() * images.size(0)  # Accumulate bbox loss for the batch

        # Compute average training loss for the epoch
        train_class_loss /= len(train_loader.dataset)
        train_bbox_loss /= len(train_loader.dataset)
        train_class_losses.append(train_class_loss)
        train_bbox_losses.append(train_bbox_loss)

        # Validation loop
        model.eval()
        with torch.no_grad():
            for images, class_labels, bbox_labels in val_loader:
                images = images.to(device)
                class_labels = class_labels.to(device)
                bbox_labels = bbox_labels.to(device)

                # Forward pass
                class_preds, bbox_preds = model(images)

                # Calculate losses
                classification_loss = class_loss_fn(class_preds, class_labels)
                bbox_regression_loss = bbox_loss_fn(bbox_preds, bbox_labels)

                val_class_loss += classification_loss.item() * images.size(0)
                val_bbox_loss += bbox_regression_loss.item() * images.size(0)

        # Compute average validation loss for the epoch
        val_class_loss /= len(val_loader.dataset)
        val_bbox_loss /= len(val_loader.dataset)
        val_class_losses.append(val_class_loss)
        val_bbox_losses.append(val_bbox_loss)

        # Apply the weight to the validation bounding box loss and calculate total validation loss
        weighted_val_loss = alpha * val_class_loss + (1 - alpha) * val_bbox_loss * 100

        # Early Stopping Check
        if weighted_val_loss < best_val_loss:
            best_val_loss = weighted_val_loss
            early_stop_counter = 0  # Reset counter if we get an improvement
             # Save the model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved at epoch {epoch+1} with validation loss: {weighted_val_loss:.4f}')
        else:
            early_stop_counter += 1
            print(f'Early stopping counter: {early_stop_counter}/{patience}')

        # Print epoch losses
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Class Loss: {train_class_loss:.4f}, Train BBox Loss: {train_bbox_loss:.4f}, Val Class Loss: {val_class_loss:.4f}, Val BBox Loss: {val_bbox_loss:.4f}')

        # Stop training if patience is exceeded
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    return train_class_losses, train_bbox_losses, val_class_losses, val_bbox_losses

def plot_losses(train_class_losses, train_bbox_losses, val_class_losses, val_bbox_losses):
    """Plots the training and validation classification and bounding box regression losses over epochs"""
    epochs = range(1, len(train_class_losses) + 1)
    plt.figure(figsize=(10, 5))

    # Plot training losses
    plt.plot(epochs, train_class_losses, 'b', label='Training Classification Loss', linestyle='--')
    plt.plot(epochs, train_bbox_losses, 'g', label='Training BBox Loss', linestyle='--')

    # Plot validation losses
    plt.plot(epochs, val_class_losses, 'r', label='Validation Classification Loss', linestyle='-')
    plt.plot(epochs, val_bbox_losses, 'orange', label='Validation BBox Loss', linestyle='-')

    # Set plot labels and title
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_iou(box1, box2):
    """Helper function to compute IoU between two bounding boxes"""
    # Convert [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Compute the area of intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute the area of both the predicted box and the ground truth box
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Compute the intersection over union (IoU)
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou

def convert_bbox_format(bbox):
    """Converts (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)."""
    x_center, y_center, width, height = bbox
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return [x_min, y_min, x_max, y_max]

def evaluate_model_with_metrics(model, data_loader, device, iou_threshold=0.5, num_classes=3):
    """Function to evaluate the model and calculate precision, recall, IoU, confusion matrix, and mAP"""
    model.eval()
    all_class_labels = []
    all_class_preds = []
    all_bbox_labels = []
    all_bbox_preds = []
    ious = []

    # Initialize mAP, precision, and recall metrics using torchmetrics and move them to the appropriate device
    mAP_metric = MeanAveragePrecision(iou_thresholds=[0.5]).to(device)
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)

    for images, class_labels, bbox_labels in data_loader:
        images = images.to(device)
        class_labels = class_labels.to(device)
        bbox_labels = bbox_labels.to(device)

        with torch.no_grad():
            class_preds, bbox_preds = model(images)
            class_preds = torch.argmax(class_preds, dim=1)

            all_class_labels.append(class_labels.cpu().numpy())
            all_class_preds.append(class_preds.cpu().numpy())

            all_bbox_labels.append(bbox_labels.cpu().numpy())
            all_bbox_preds.append(bbox_preds.cpu().numpy())

            # Update mAP metric with predictions and labels
            for i in range(len(bbox_preds)):
                # Convert from (x_center, y_center, w, h) to (x_min, y_min, x_max, y_max)
                pred_box = convert_bbox_format(bbox_preds[i].cpu().numpy())
                target_box = convert_bbox_format(bbox_labels[i].cpu().numpy())

                # Extract confidence score from the model's predictions (assuming sigmoid for bbox output)
                confidence_score = torch.sigmoid(class_preds[i].float()).max().item()

                preds = [{
                    'boxes': torch.tensor([pred_box]),
                    'scores': torch.tensor([confidence_score]),
                    'labels': class_preds[i].unsqueeze(0).cpu()
                }]
                targets = [{
                    'boxes': torch.tensor([target_box]),
                    'labels': class_labels[i].unsqueeze(0).cpu()
                }]
                mAP_metric.update(preds, targets)

            # Update precision and recall metrics
            precision_metric.update(class_preds, class_labels)
            recall_metric.update(class_preds, class_labels)

            # Calculate IoU for the current batch
            for bbox_pred, bbox_label in zip(bbox_preds, bbox_labels):
                pred_box = convert_bbox_format(bbox_pred.cpu().numpy())
                target_box = convert_bbox_format(bbox_label.cpu().numpy())
                iou = calculate_iou(pred_box, target_box)
                ious.append(iou)

    # Calculate confusion matrix
    all_class_labels = np.concatenate(all_class_labels)
    all_class_preds = np.concatenate(all_class_preds)
    conf_matrix = confusion_matrix(all_class_labels, all_class_preds)

    # Compute the mAP from the accumulated data
    mAP_result = mAP_metric.compute()

    # Compute precision and recall
    precision = precision_metric.compute()
    recall = recall_metric.compute()

    # Calculate mean IoU
    mean_iou = np.mean([ious])

    return precision, recall, conf_matrix, mAP_result, mean_iou

def show_predictions_one_by_one(model, dataset, classes, device, num_images=30):
    """Helper function to plot bounding boxes and metrics on an image, one by one"""
    model.eval()  # Set model to evaluation mode

    for i in range(num_images):
        # Get a random image and its ground truth from the dataset
        image, true_class, true_bbox = dataset[i]
        image = image.unsqueeze(0).to(device)  # Add batch dimension and send to device
        true_class = true_class.item()
        true_bbox = true_bbox.cpu().numpy()

        # Get model predictions
        with torch.no_grad():
            class_preds, bbox_preds = model(image)
            predicted_class = torch.argmax(class_preds, dim=1).cpu().numpy()[0]
            predicted_bbox = bbox_preds.cpu().numpy()[0]

        # Convert the image tensor to numpy for visualization
        image_np = image.squeeze().cpu().permute(1, 2, 0).numpy()  # Remove batch dimension and convert to HWC

        # Calculate IoU
        iou = calculate_iou(predicted_bbox, true_bbox)

        # Plot the image
        plt.figure(figsize=(10, 5))
        plt.imshow(image_np)

        # Ground truth bounding box
        img_height, img_width = image_np.shape[:2]
        x_center, y_center, box_width, box_height = true_bbox
        x1_true = (x_center - box_width / 2) * img_width
        y1_true = (y_center - box_height / 2) * img_height
        rect_true = patches.Rectangle((x1_true, y1_true), box_width * img_width, box_height * img_height,
                                      linewidth=2, edgecolor='green', facecolor='none', label="Ground Truth")
        plt.gca().add_patch(rect_true)

        # Predicted bounding box
        x_center_pred, y_center_pred, box_width_pred, box_height_pred = predicted_bbox
        x1_pred = (x_center_pred - box_width_pred / 2) * img_width
        y1_pred = (y_center_pred - box_width_pred / 2) * img_height
        rect_pred = patches.Rectangle((x1_pred, y1_pred), box_width_pred * img_width, box_height_pred * img_height,
                                      linewidth=2, edgecolor='red', facecolor='none', label="Prediction")
        plt.gca().add_patch(rect_pred)

        plt.title(f"True: {classes[true_class]}\nPred: {classes[predicted_class]}\nIoU: {iou:.2f}")

        plt.show()

class ObjectDetectionDataset(Dataset):
    """
    Custom PyTorch Dataset for object detection.
    Loads images and corresponding labels from a specified directory.
    Labels include class index and bounding box coordinates.
    """

    def __init__(self, root_dir, split='train', num_classes=3, transform=None):
        """Initializes the dataset with paths to image and label directories. Optionally applies transformations to the images."""
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.split_dir, 'images')
        self.label_dir = os.path.join(self.split_dir, 'labels')
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Loads and returns a single image and its corresponding label (class and bounding box). Applies transformation if specified."""
        # Load image
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Load label
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            label = f.readline().strip().split()
            category = int(label[0])
            bbox = [float(x) for x in label[1:]]

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(category), torch.tensor(bbox)

class DetectionModel(nn.Module):
    """
    Object detection model based on a ResNet50 backbone with a Feature Pyramid Network (FPN).
    Outputs class predictions and bounding box coordinates.
    """

    def __init__(self, num_classes=3, pretrained=True):
        """Initializes the detection model with ResNet50 backbone and FPN. Defines separate branches for classification and bounding box regression."""
        super().__init__()

        # Backbone with feature extraction
        backbone = resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels

        # Feature Pyramid Network
        self.fpn_lateral4 = nn.Conv2d(2048, 256, 1)
        self.fpn_lateral3 = nn.Conv2d(1024, 256, 1)
        self.fpn_lateral2 = nn.Conv2d(512, 256, 1)

        self.fpn_output4 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_output3 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_output2 = nn.Conv2d(256, 256, 3, padding=1)

        # Heads
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )  # Закрывающая скобка была пропущена

        self.bbox_regressor = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 4),
            nn.Sigmoid())

    def forward(self, x):
        """Defines the forward pass through the backbone, FPN, and prediction heads. Returns class scores and bounding box predictions."""
        # Bottom-up pathway
        c1 = self.layer0(x)
        c2 = self.layer1(c1)  # 1/4
        c3 = self.layer2(c2)  # 1/8
        c4 = self.layer3(c3)  # 1/16
        c5 = self.layer4(c4)  # 1/32

        # Top-down pathway and lateral connections
        p5 = self.fpn_lateral4(c5)
        p4 = self.fpn_lateral3(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.fpn_lateral2(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')

        # FPN outputs
        p5 = self.fpn_output4(p5)
        p4 = self.fpn_output4(p4)
        p3 = self.fpn_output3(p3)

        # Use the best feature level (p3 for medium-sized objects)
        class_logits = self.classifier(p3)
        bbox_coords = self.bbox_regressor(p3)

        return class_logits, bbox_coords
    
class CustomModel(nn.Module):
    """
    A custom object detection model with a deep CNN backbone.
    Includes separate branches for classification and bounding box regression.
    Designed for flexibility and experimentation.
    """

    def __init__(self, num_classes=3):
        """Initializes a custom convolutional neural network for object detection. Builds the backbone, classifier, and bounding box regressor."""
        super(CustomModel, self).__init__()

        # Define custom backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classification branch with Dropout
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.4),  # Add dropout for regularization
            nn.Linear(1024, num_classes)    # Linear layer to output class probabilities
        )

        # Bounding box regression branch aligned with backbone's final output
        self.bbox_regressor = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1), # Use 2048 as input from backbone
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.3), # Add dropout for regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4) # Linear layer to output 4 bounding box coordinates
        )

    def forward(self, x):
        """Defines the forward pass of the model. Extracts features with the backbone and passes them to both the classification and regression heads."""
        # Shared backbone
        features = self.backbone(x)

        # Classification branch
        class_output = self.classifier(features)

        # Bounding box regression branch
        bbox_output = self.bbox_regressor(features)

        return class_output, bbox_output
