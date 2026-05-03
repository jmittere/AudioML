import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from mel_dataset import collate_fn
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mel(model_type, model, train_dataset, val_dataset, num_epochs=50, batch_size=4, lr=1e-4, output_dir="../outputs", patience=7, min_delta=1e-5, save_path=None):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    model = model.to(device)
    print("Device: ", next(model.parameters()).device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, threshold=min_delta)
    criterion = torch.nn.MSELoss()
    train_losses = []
    val_losses = []

    best_val_loss = float(999999)
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0

        progress = tqdm(train_dataloader,desc=f"Epoch {epoch}/{num_epochs}",leave=False)

        for mel, _ in progress:
            if model_type == "MelTransformerFrameBin":
                loss = train_step_framebin(model, criterion, optimizer, mel)
            elif model_type == "MelTransformerFrame":
                loss = train_step_frame(model, criterion, optimizer, mel)
            elif model_type == "MelTransformerFrameDelta":
                loss = train_step_frame_delta(model, criterion, optimizer, mel)
            total_loss += loss
            progress.set_postfix(train_loss=f"{loss:.4f}")

        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        if model_type == "MelTransformerFrameBin":
            avg_val_loss = validate_framebin(model, val_dataloader, criterion)
        elif model_type == "MelTransformerFrame":
            avg_val_loss = validate_frame(model, val_dataloader, criterion)
        elif model_type == "MelTransformerFrameDelta":
            avg_val_loss = validate_frame_delta(model, val_dataloader, criterion)

        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: avg_train_loss = {avg_loss:.4f}, avg_val_loss = {avg_val_loss:.4f}, Current LR: {current_lr:.6f}")
        #Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                #print(f"saved best model to {save_path}")

        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        #Stop training
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break


    # Plot the loss curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.title("Training and Validation Loss")
    plt.savefig(f"{output_dir}/training vs val loss.png", dpi=300, bbox_inches="tight")
    plt.close()

###-- training functions for frame*freq_bin autoregression --###
def train_step_framebin(model, criterion, optimizer, mel):

    mel = mel.to(device)

    B, T, F = mel.shape

    mel_flat = mel.reshape(B, T * F, 1)

    #Shift in flattened space, still using autoregressive training but now with prev freq frames too
    x_input  = mel_flat[:, :-1, :]   # (B, L-1, 1)
    x_target = mel_flat[:, 1:, :]    # (B, L-1, 1)

    preds = model(x_input)           # (B, L-1, 1)

    optimizer.zero_grad()
    loss = criterion(preds, x_target)
    loss.backward()
    optimizer.step()

    return loss.item()

def validate_framebin(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for mel,_ in dataloader:
            mel = mel.to(device)

            B, T, F = mel.shape

            mel_flat = mel.reshape(B, T * F, 1)

            x_input  = mel_flat[:, :-1, :]
            x_target = mel_flat[:, 1:, :]

            preds = model(x_input)

            loss = criterion(preds, x_target)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_framebin(model, dataset, val_indices, seed_seconds=7.0, sr=22050, hop_length=256):
    model.eval()

    results=[]

    for i in val_indices:
        mel, filepath = dataset[i]  # (T, F)
        mel = mel.to(device)

        seed_frames = int(seed_seconds * sr / hop_length)

        seed = mel[:seed_frames].unsqueeze(0)  # (1, T_seed, F)

        baseline = get_repeated_last_frame_baseline(mel, seed, seed_frames)

        #flatten
        generated = seed.reshape(1, -1, 1)

        total_bins = mel.shape[0] * mel.shape[1]
        seed_bins = generated.shape[1]

        num_future = total_bins - seed_bins

        with torch.no_grad():
            for _ in range(num_future):
                preds = model(generated)
                next_bin = preds[:, -1:, :]
                generated = torch.cat([generated, next_bin], dim=1)

        #reshape back
        generated = generated.reshape(mel.shape[0], mel.shape[1])
        metrics = evaluate_preds(mel.cpu(),generated,baseline.cpu(),seed_frames)
        results.append((mel.cpu(),generated,baseline.cpu(),filepath,metrics))

    return results

#--------------------------------------------------------------#

###-- training functions for full frame autoregression --###
def train_step_frame(model, criterion, optimizer, mel,scheduled_sampling_prob=0.15):

    mel = mel.to(device)

    # shift for autoregressive training
    x_input  = mel[:, :-1, :]   # (B, T-1, 80)
    x_target = mel[:, 1:, :]    # (B, T-1, 80)

    B, T, _ = x_input.shape
    with torch.no_grad():
        preds = model(x_input)

    #Create mask for scheduled sampling
    mask = torch.rand(B, T, 1, device=x_input.device) < scheduled_sampling_prob

    #Mix ground truth and predictions
    x_mixed = torch.where(mask, preds, x_input)

    #Forward pass
    preds = model(x_mixed)

    optimizer.zero_grad()
    loss = criterion(preds, x_target)
    loss.backward()
    optimizer.step()

    return loss.item()

def validate_frame(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for mel, _ in dataloader:
            mel = mel.to(device)

            x_input  = mel[:, :-1, :]
            x_target = mel[:, 1:, :]

            preds = model(x_input)

            loss = criterion(preds, x_target)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_frame(model, dataset, seed_seconds, val_indices, sr=22050, hop_length=256):
    model.eval()

    results = []

    for i in val_indices:

        mel, filepath = dataset[i]  # (T, 80)
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
        baseline = get_repeated_last_frame_baseline(mel, seed, seed_frames)
        metrics = evaluate_preds(mel.cpu(),generated,baseline.cpu(),seed_frames)
        results.append((mel.cpu(),generated,baseline.cpu(),filepath,metrics))

    return results

#--------------------------------------------------------------#

###-- training functions for full frame delta autoregression --###
def train_step_frame_delta(model, criterion, optimizer, mel):

    mel = mel.to(device)

    #shift
    x_input  = mel[:, :-1, :]   # (B, T-1, F)
    x_target = mel[:, 1:, :]    # (B, T-1, F)

    #residual target
    delta_target = x_target - x_input

    #Forward pass
    delta_pred = model(x_input)

    optimizer.zero_grad()
    loss = criterion(delta_pred, delta_target)
    loss.backward()
    optimizer.step()

    return loss.item()

def validate_frame_delta(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for mel, _ in dataloader:
            mel = mel.to(device)

            x_input  = mel[:, :-1, :]
            x_target = mel[:, 1:, :]

            delta_target = x_target - x_input
            delta_pred = model(x_input)

            loss = criterion(delta_pred, delta_target)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_frame_delta(model, dataset, seed_seconds, val_indices, sr=22050, hop_length=256):
    model.eval()

    results = []

    for i in val_indices:

        mel, filepath = dataset[i]
        mel = mel.to(device)

        seed_frames = int(seed_seconds * sr / hop_length)
        seed = mel[:seed_frames].unsqueeze(0)

        generated = seed.clone()

        num_future = mel.shape[0] - seed_frames

        with torch.no_grad():
            for _ in range(num_future):
                delta = model(generated) # (1, T, F)
                delta_last = delta[:, -1:, :] #last timestep delta, difference between xt-1 and xt
                last_frame = generated[:, -1:, :]
                next_frame = last_frame + delta_last
                generated = torch.cat([generated, next_frame], dim=1)

        generated = generated.squeeze(0).cpu()
        baseline = get_repeated_last_frame_baseline(mel, seed, seed_frames)
        metrics = evaluate_preds(mel.cpu(),generated,baseline.cpu(),seed_frames)
        results.append((mel.cpu(), generated, baseline.cpu(), filepath, metrics))

    return results


def get_repeated_last_frame_baseline(mel, seed, seed_frames):
    num_future_frames = mel.shape[0] - seed_frames
    #get last timestep for all frequencies in seed frame
    last_frame = seed[:, -1:, :] #(1,1,n_mels)

    #repeat for future timesteps
    repeated_frames = last_frame.repeat(1,num_future_frames, 1) #(1,num_future,n_mels)
    baseline = torch.cat([seed, repeated_frames], dim=1) #stitch seed and repeated last frame baseline
    baseline = baseline.squeeze(0)
    return baseline

def evaluate_preds(gt, pred, baseline, seed_frames):
    gt = gt.cpu()
    pred = pred.cpu()
    baseline = baseline.cpu()
    gt_future = gt[seed_frames:]
    pred_future = pred[seed_frames:]
    baseline_future = baseline[seed_frames:]

    mse_model = torch.mean((pred_future - gt_future) ** 2)
    mse_baseline = torch.mean((baseline_future - gt_future) ** 2)

    mae_model = torch.mean(torch.abs(pred_future - gt_future))
    mae_baseline = torch.mean(torch.abs(baseline_future - gt_future))

    improvement = (mse_baseline - mse_model) / (mse_baseline + 1e-8)

    return {
        "mse_model": mse_model.item(),
        "mse_baseline": mse_baseline.item(),
        "mae_model": mae_model.item(),
        "mae_baseline": mae_baseline.item(),
        "improvement": improvement.item()
    }
