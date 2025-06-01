import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model.pipeline.preparation import prepare_data
from config.config import settings
from loguru import logger
from db.db_model import SignMnistTrain, SignMnistTest


def build_model():
    """
    Builds and trains the CNN model for sign language recognition.
    This function prepares the data, splits it into training and
    validation sets, creates DataLoaders, trains the model,
    evaluates it on the test set, and saves the model.
    """
    logger.info("Building the CNN model")
    # Prepare the data
    train_data = prepare_data(SignMnistTrain)
    test_data = prepare_data(SignMnistTest)

    # Split the training data into training and validation sets
    X_train, X_valid, y_train, y_valid = split_data(train_data[0],
                                                    train_data[1])
    X_test, y_test = test_data

    # Create DataLoaders for training, validation, and test sets
    train_loader = create_dataloader(X_train, y_train)
    valid_loader = create_dataloader(X_valid, y_valid)
    test_loader = create_dataloader(X_test, y_test)

    # Train the model
    model = train_model(train_loader, valid_loader, epochs=25,
                        learning_rate=0.001)

    # Evaluate the model on the test set
    eval_loss, eval_acc = evaluate_model(model, test_loader)

    # Save the model
    save_model(model)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
                               padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                               padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 26)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.reshape(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def split_data(X, y, train_size=0.8):
    """
    Splits the data into training and validation sets.
    """
    logger.info("Splitting data into training and validation sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=train_size,
                                                        random_state=12345)
    return X_train, X_test, y_train, y_test


def create_dataloader(X, y, batch_size=512):
    """
    Creates a DataLoader for the given data.
    """
    logger.info("Creating DataLoader for the data")
    if torch.cuda.is_available():
        X = X.to('cuda')
        y = y.to('cuda')
    else:
        X = X.to('mps')
        y = y.to('mps')
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_model(train_loader, valid_loader, epochs=10, learning_rate=0.005):
    """
    Trains the CNN model on the training data and
    validates it on the validation data.
    """
    logger.info("Starting model training")
    model = CNNModel()
    model = model.to('cuda' if torch.cuda.is_available() else 'mps')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss/len(train_loader)
        train_acc = 100 * correct/total
        logger.info(f"Epoch [{epoch+1}/{epochs}],\
                    Train Loss: {train_loss:.4f},\
                    Train Accuracy: {train_acc:.2f}%")

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.permute(0, 3, 1, 2)
                outputs = model(inputs)
                labels = labels.long()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss/len(valid_loader)
        val_acc = 100*correct/total
        logger.info(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f},\
                    Val Accuracy: {val_acc:.2f}%")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
    logger.info("Model training completed")
    return model


def evaluate_model(model, test_loader):
    """
    Evaluates the trained model on the test data.
    """
    logger.info("Evaluating model on test data")
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        for inputs, labels in test_loader:
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss = test_loss/len(test_loader)
        test_acc = 100*correct/total
        logger.info(f"Test Loss: {test_loss:.4f},\
                    Test Accuracy: {test_acc:.2f}%")
    return test_loss, test_acc


def save_model(model,
               file_path=f'{settings.model_save_path}/{settings.model_name}'):
    """
    Saves the trained model to a file.
    """
    logger.info(f"Saving model to {file_path}")
    torch.save(model.state_dict(), file_path)
