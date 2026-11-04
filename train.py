import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import get_dataloader
from model import DDPM_MLP
from diffusion import Diffusion
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use("default")
plt.style.use(hep.style.CMS)
plt.rcParams['font.size'] = 12


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up mode, scaler and paths
    scaler_path = args.scaler_path if args.scaler_path else f"scaler_{args.mode}_{args.scaler}.pkl"
    ckpt_prefix = args.checkpoint_prefix if args.checkpoint_prefix else f"ddpm_{args.mode}_{args.scaler}"
    feature_dim = 8 if args.mode == 'cartesian' else 6

    print(f"============================================================")
    print(f" Training Configuration:")
    print(f"   Mode:              {args.mode} (Dimension: {feature_dim})")
    print(f"   Scaler:            {args.scaler}")
    print(f"   Scaler Path:       {scaler_path}")
    print(f"   Checkpoint Prefix: {ckpt_prefix}")
    print(f"   Train Split:       {args.train_split*100:.0f}% train / {(1-args.train_split)*100:.0f}% test (seed: {args.seed})")
    print(f"   Epochs:            {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"============================================================")

    # DataLoader (train split)
    dataloader = get_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        shuffle=True,
        fit_scaler=True,
        scaler_path=scaler_path,
        mode=args.mode,
        scaler_type=args.scaler,
        split='train',
        train_ratio=args.train_split,
        seed=args.seed
    )

    # Model & Diffusion
    model = DDPM_MLP(feature_dim=feature_dim, hidden_dim=256, time_emb_dim=256, num_layers=4).to(device)
    diffusion = Diffusion(timesteps=args.timesteps, device=device, schedule='linear')

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.MSELoss()

    # Optionally resume from a checkpoint
    start_epoch = 0
    losses = []
    os.makedirs('checkpoints', exist_ok=True)
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        losses = checkpoint.get('losses', [])
        print(f"Resumed from epoch {start_epoch} (checkpoint: {args.resume})")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"[{args.mode}|{args.scaler}] Epoch {epoch+1}/{start_epoch+args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device, dtype=torch.long)

            # Forward diffusion process (add noise)
            x_noisy, noise = diffusion.forward_diffusion_sample(batch, t)

            # Predict noise
            predicted_noise = model(x_noisy, t)

            # Loss and optimize
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{start_epoch+args.epochs} Average Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        scheduler.step()

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = f'checkpoints/{ckpt_prefix}_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'losses': losses,
                'mode': args.mode,
                'scaler': args.scaler,
                'feature_dim': feature_dim
            }, ckpt_path)
            torch.save(model.state_dict(), ckpt_path.replace('.pt', '_weights.pt'))

    # Save final model
    final_weights = f"checkpoints/{ckpt_prefix}_final.pt"
    torch.save(model.state_dict(), final_weights)
    torch.save(model.state_dict(), "checkpoints/ddpm_final.pt")  # alias default

    torch.save({
        'epoch': start_epoch + args.epochs - 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'losses': losses,
        'mode': args.mode,
        'scaler': args.scaler,
        'feature_dim': feature_dim
    }, f'checkpoints/{ckpt_prefix}_final_checkpoint.pt')

    # Plot loss
    loss_fig = f'loss_plot_{args.mode}_{args.scaler}.png'
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, label=f'Loss ({args.mode} + {args.scaler})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'DDPM Loss ({args.mode} | {args.scaler})')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_fig)
    plt.savefig('loss_plot.png')
    plt.close()

    print(f"\nTraining complete.")
    print(f"Model saved to: {final_weights}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Loss plot saved to: {loss_fig}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DDPM for D0 kinematics")
    parser.add_argument('--data_path', type=str, default='data_D0_daughtersPx_Py_Pz_E_trk1_trk2.txt', help='Path to input dataset')
    parser.add_argument('--mode', type=str, default='rest_frame', choices=['rest_frame', 'cartesian'], help="Input representation: 'rest_frame' (6D) or 'cartesian' (8D)")
    parser.add_argument('--scaler', type=str, default='quantile', choices=['quantile', 'standard'], help="Scaler: 'quantile' (QuantileTransformer) or 'standard' (StandardScaler)")
    parser.add_argument('--scaler_path', type=str, default=None, help="Explicit scaler save path (default: scaler_{mode}_{scaler}.pkl)")
    parser.add_argument('--checkpoint_prefix', type=str, default=None, help="Prefix for saved checkpoints (default: ddpm_{mode}_{scaler})")
    parser.add_argument('--train_split', type=float, default=0.8, help="Train split fraction (default: 0.8 = 80%% train / 20%% test)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for train/test split")
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    args = parser.parse_args()

    train(args)
