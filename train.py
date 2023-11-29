import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from torch.utils.data import DataLoader
from dataset import iemocap

logging.basicConfig(filename='training.log', level=logging.INFO)

# Import your dataset and model here

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def main(args):
    # Define device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    # Load your dataset using DataLoader
    # dataset = YourDataset(...)  # You should define YourDataset elsewhere
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataset = iemocap.IEMOCAP()
    # Define your model
    # model = YourModel(...)  # You should define YourModel elsewhere
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Optionally resume training from a checkpoint
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
    else:
        start_epoch = 0
        best_loss = float('inf')

    for epoch in range(args.num_epochs):
        train_loss = train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {train_loss:.4f}")
        logging.info(f"Epoch [{epoch + 1}/{args.num_epochs}] - Train Loss: {train_loss:.4f}")

        # Save model checkpoint if necessary
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f'model_checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)

            # Optionally update best_loss if the current loss is better
            if train_loss < best_loss:
                best_loss = train_loss
                best_checkpoint_path = checkpoint_path

    # Save the final trained model
    final_model_path = 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)

    # Optionally log the path to the best checkpoint
    if args.resume_checkpoint:
        print(f"Training completed. Best checkpoint at: {best_checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training Script")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--save-interval", type=int, default=1, help="Interval for saving model checkpoints")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to resume training from a checkpoint")

    args = parser.parse_args()
    main(args)