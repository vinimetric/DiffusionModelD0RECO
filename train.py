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

# Feature dimension: pt_K, y_K, phi_K, pt_pi, y_pi, phi_pi
FEATURE_DIM = 6

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # DataLoader
    dataloader = get_dataloader(
        args.data_path, 
        batch_size=args.batch_size, 
        shuffle=True, 
        fit_scaler=True, 
        scaler_path='scaler.pkl'
    )
    
    # Model & Diffusion
    model = DDPM_MLP(feature_dim=FEATURE_DIM, hidden_dim=256, time_emb_dim=256, num_layers=4).to(device)
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
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch+args.epochs}")
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
            ckpt_path = f'checkpoints/ddpm_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'losses': losses,
            }, ckpt_path)
            # Also save the plain weights for generate.py
            torch.save(model.state_dict(), ckpt_path.replace('.pt', '_weights.pt'))
            
    # Save final model
    torch.save(model.state_dict(), "checkpoints/ddpm_final.pt")
    torch.save({
        'epoch': start_epoch + args.epochs - 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'losses': losses,
    }, 'checkpoints/ddpm_final_checkpoint.pt')
    
    # Plot loss
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()
    
    print("Training complete. Models saved to checkpoints/, loss plot saved to loss_plot.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,  help='Path to data_D0_daughtersPx_Py_Pz_E_trk1_trk2.txt', default='data_D0_daughtersPx_Py_Pz_E_trk1_trk2.txt')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    train(args)
