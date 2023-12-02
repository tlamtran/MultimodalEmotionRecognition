import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Processor, RobertaTokenizer, VivitImageProcessor

audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
text_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

def collate_fn(batch):
    audio_inputs, text_inputs, video_inputs, labels = zip(*batch)

    audio_inputs = audio_processor(audio_inputs, return_tensors='pt', sampling_rate=16000, padding=True)
    text_inputs = text_tokenizer(text_inputs, return_tensors='pt', padding=True)
    video_inputs = image_processor(video_inputs, return_tensors="pt", padding=True)

    return audio_inputs, text_inputs, video_inputs, torch.tensor(labels)


def train(model_name, model, train_data, val_data, epochs, batch_size, learning_rate, audio_modality, text_modality, video_modality):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()

        total_train_loss = 0
        for audio_input, text_input, video_input, targets in tqdm(train_loader):
            if audio_modality:
                audio_input = audio_input.to(device)
            if text_modality:
                text_input = text_input.to(device)
            if video_modality:
                video_input = video_input.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(audio_input, text_input, video_input)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            total_val_loss = 0; correct = 0; total_samples = 0
            for audio_input, text_input, video_input, targets in val_loader:
                audio_input = audio_input.to(device) 
                text_input = text_input.to(device)
                video_input = video_input.to(device)
                targets = targets.to(device)

                outputs = model(audio_input, text_input, video_input)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                total_samples += targets.size(0)
                correct += (outputs.argmax(dim=1) == targets).sum().item()
            val_loss = total_val_loss / len(val_loader)
            val_acc = correct / total_samples

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"{model_name}_best.pth")

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    print(f"Training completed.")
    torch.save(model.state_dict(), f"{model_name}.pth")
