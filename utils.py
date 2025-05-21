import torch
import torch.nn as nn
import time
import sys

class MyConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.model(x)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def get_trainer(model, lr, weight_decay):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=lr,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs
    # )

    return optimizer

def train(model, train_loader, loss_function, optimizer):
    model.train()
    total_loss = 0
    train_N = len(train_loader.dataset)

    for batch_idx, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_function(output, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Clear CUDA cache
        torch.cuda.empty_cache()

    return total_loss/train_N

# Validation function
def validate(model, val_loader, loss_function):
    model.eval()
    val_N = len(val_loader.dataset)
    total_loss = 0
    correct = 0
    total = 0
    # all_labels = []
    # all_predictions = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # all_labels.extend(labels.cpu().numpy())
            # all_predictions.extend(predicted.cpu().numpy())

            # Clear CUDA cache
            torch.cuda.empty_cache()

    accuracy = 100 * correct / total
    #print(classification_report(all_labels, all_predictions))
    return total_loss / val_N , accuracy

def fit_one_cycle(epochs, model, train_loader, val_loader, save_path="fruit_custom_model_4.pth"):
    cost_train = []
    cost_valid = []
    train_N = len(train_loader.dataset)
    save_model = SaveModel(save_path = save_path)

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)
    early_stopping = EarlyStopping(patience=20, min_delta=0.01)

    for epoch in range(epochs):

        train_loss = train(model, train_loader, optimizer = optimizer, loss_function = loss_function)

        valid_loss, valid_acc = validate(model, val_loader, loss_function = loss_function)

        scheduler.step(valid_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}, Validation Accuracy: {valid_acc:.2f}%, lr: {get_lr(optimizer):.6f}")
        if epoch % 10 == 0:
            cost_train.append(train_loss)
            cost_valid.append(valid_loss)
        
        save_model(valid_loss, valid_acc , model, optimizer=optimizer)

        if early_stopping(valid_loss):
            print("Early stopping triggered. Training stopped!")
            break

    plt.plot(cost_train, label='Train Loss', color='blue')
    plt.plot(cost_valid, label='Valid Loss', color='yellow')
    plt.ylabel('cost')
    plt.xlabel('epochs (per 10)')
    plt.title('Training & Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()