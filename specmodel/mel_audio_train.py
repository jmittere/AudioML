import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from mel_dataset import collate_fn, MelMaskedDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mel(model, dataset, num_epochs=50, batch_size=4, lr=1e-4):

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    model = model.to(device)
    print("Device: ", next(model.parameters()).device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0

        progress = tqdm(dataloader,desc=f"Epoch {epoch}/{num_epochs}",leave=False)

        for mel_inputs, mel_targets, mask_starts in progress:
            loss = train_step(model, criterion, optimizer, mel_inputs, mel_targets, mask_starts)
            total_loss += loss
            progress.set_postfix(loss=f"{loss:.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")


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

def eval(model, dataset):
    model.eval()
    mel_input, mel_target, mask_start = dataset[0]
    mel_input = mel_input.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(mel_input)

    pred_tail = pred[0, mask_start:].cpu().numpy()  # (mask_T, 80)