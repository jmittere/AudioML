import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from mel_dataset import collate_fn, MelMaskedDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mel(model, train_dataset, val_dataset, num_epochs=50, batch_size=4, lr=1e-4):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)


    model = model.to(device)
    print("Device: ", next(model.parameters()).device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0

        progress = tqdm(train_dataloader,desc=f"Epoch {epoch}/{num_epochs}",leave=False)

        for mel_inputs, mel_targets, mask_starts in progress:
            loss = train_step(model, criterion, optimizer, mel_inputs, mel_targets, mask_starts)
            total_loss += loss
            progress.set_postfix(train_loss=f"{loss:.4f}")

        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        avg_val_loss = validate(model, val_dataloader, criterion)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch}: avg_train_loss = {avg_loss:.4f}, avg_val_loss = {avg_val_loss:.4f}")


    # Plot the loss curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper left")
    plt.title('Training and Validation Loss')
    plt.savefig("training vs val loss.png", dpi=300, bbox_inches="tight")
    plt.close()

def train_step(model, criterion, optimizer, mel_inputs, mel_targets, mask_starts):
    
    mel_inputs = mel_inputs.to(device)
    mel_targets = mel_targets.to(device)
    preds = model(mel_inputs)  # (B, T, 80)

    # Collect predictions for masked region only
    pred_tail = []
    for b in range(preds.shape[0]):
        start = mask_starts[b].item()
        pred_tail.append(preds[b, start:start + mel_targets.shape[1]])

    pred_tail = torch.stack(pred_tail)

    loss = criterion(pred_tail, mel_targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for mel_inputs, mel_targets, mask_starts in dataloader:
            mel_inputs = mel_inputs.to(device)
            mel_targets = mel_targets.to(device)

            preds = model(mel_inputs)

            pred_tail = []
            for b in range(preds.shape[0]):
                start = mask_starts[b].item()
                pred_tail.append(preds[b, start:start + mel_targets.shape[1]])

            pred_tail = torch.stack(pred_tail)
            loss = criterion(pred_tail, mel_targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def eval(model, dataset):
    model.eval()
    mel_input, mel_target, mask_start, filepath = dataset[0]
    print("filepath: ", filepath)
    print("Mel_inputs type:" , type(mel_input))
    print("Mel_inputs.shape:" , mel_input.shape)
    print("Mel_targets type:" , type(mel_target))
    print("Mel targets.shape: ", mel_target.shape)
    print("mask starts type:" , type(mask_start))
    print("mask starts: ", mask_start)
    ground_truth_mel = torch.vstack((mel_input[0:mask_start, ], mel_target))
    print("ground_truth_mel:" , type(ground_truth_mel))
    print(ground_truth_mel.shape)
    print(ground_truth_mel)
    
    mel_input = mel_input.unsqueeze(0).to(device)
    print("Mel_inputs.shape:" , mel_input.shape)

    with torch.no_grad():
        pred = model(mel_input)

    #pred_tail = pred[0, mask_start:].cpu().numpy()  # (mask_T, 80)
    mel_input.to('cpu')
    pred_tail = pred[0, mask_start:]  # (mask_T, 80)
    mel_input = mel_input.squeeze(dim=0)
    full_mel_pred = torch.vstack((mel_input[0:mask_start, ], pred_tail))
    print("full_mel_pred:" , type(full_mel_pred))
    print(full_mel_pred.shape)
    print(full_mel_pred)
    return ground_truth_mel, full_mel_pred
