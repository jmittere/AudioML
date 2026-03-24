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
    criterion = torch.nn.MSELoss()
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0

        progress = tqdm(train_dataloader,desc=f"Epoch {epoch}/{num_epochs}",leave=False)

        for mel in progress:
            loss = train_step(model, criterion, optimizer, mel)
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

def train_step(model, criterion, optimizer, mel):

    mel = mel.to(device)

    # shift for autoregressive training
    x_input  = mel[:, :-1, :]   # (B, T-1, 80)
    x_target = mel[:, 1:, :]    # (B, T-1, 80)

    use_model_pred = torch.rand(1).item() < 0.1

    #scheduled sampling randomly
    if use_model_pred:
        with torch.no_grad():
            preds = model(x_input)
        x_input = preds.detach()
        

    preds = model(x_input)

    optimizer.zero_grad()
    loss = criterion(preds, x_target)
    loss.backward()
    optimizer.step()

    return loss.item()

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for mel in dataloader:
            mel = mel.to(device)

            x_input  = mel[:, :-1, :]
            x_target = mel[:, 1:, :]
            

            preds = model(x_input)

            loss = criterion(preds, x_target)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def eval(model, dataset, seed_seconds=7.0, sr=22050, hop_length=256):
    model.eval()

    mel = dataset[0]  # (T, 80)
    mel = mel.to(device)

    seed_frames = int(seed_seconds * sr / hop_length)

    seed = mel[:seed_frames].unsqueeze(0)  # (1, T_seed, 80)

    generated = seed.clone()

    num_future = mel.shape[0] - seed_frames

    with torch.no_grad():
        for _ in range(num_future):
            preds = model(generated)

            next_frame = preds[:, -1:, :]  # last timestep

            generated = torch.cat([generated, next_frame], dim=1)

    generated = generated.squeeze(0).cpu()

    return mel.cpu(), generated