import torch

import math

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class Diffusion:
    def __init__(self, timesteps=1000, device='cpu', schedule='cosine'):
        self.timesteps = timesteps
        self.device = device
        
        if schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps).to(device)
        elif schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t):
        """
        Takes an original sample x_0 and a timestep t
        and returns the noisy version of x_0 at t.
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # mean + variance
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    @torch.no_grad()
    def sample_timestep(self, model, x, t):
        """
        Calls the model to predict the noise in the sample and returns 
        the denoised sample at t-1.
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(self.get_index_from_list(self.alphas, t, x.shape))
        
        # Call model (current sample x, timestep t)
        predicted_noise = model(x, t)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0].item() == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t # simpler variance assumption
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    @torch.no_grad()
    def sample(self, model, n_samples, feature_dim=8):
        """
        Samples n_samples from the model using the standard DDPM stochastic reverse process.
        """
        model.eval()
        
        # Start from pure noise
        x = torch.randn((n_samples, feature_dim), device=self.device)
        
        for i in reversed(range(0, self.timesteps)):
            # The timestep needs to be batched
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            # Denoise step
            x = self.sample_timestep(model, x, t)
            
        return x

    @torch.no_grad()
    def sample_ddim(self, model, n_samples, feature_dim=8, ddim_steps=200, eta=0.0):
        """
        Samples using the DDIM (Song et al. 2020) deterministic reverse process.
        
        Key advantages over DDPM sampling:
        - Deterministic with eta=0: no stochastic overshooting of distribution modes
        - Can use far fewer steps (200 vs 1000) without quality loss
        - Smoother output distributions with fewer artifacts
        
        Args:
            eta: stochasticity parameter. eta=0 -> fully deterministic (DDIM).
                 eta=1 -> equivalent to DDPM. Values 0 < eta < 1 give a blend.
        """
        model.eval()
        
        # Select subset of timesteps (evenly spaced)
        step_ratio = self.timesteps // ddim_steps
        timesteps_seq = list(reversed(range(0, self.timesteps, step_ratio)))  # [T-1, ..., 0]
        
        # Start from pure noise
        x = torch.randn((n_samples, feature_dim), device=self.device)
        
        for i, t_val in enumerate(timesteps_seq):
            t = torch.full((n_samples,), t_val, device=self.device, dtype=torch.long)
            
            # Previous timestep in the subsequence
            t_prev_val = timesteps_seq[i + 1] if i + 1 < len(timesteps_seq) else -1
            
            # Get cumulative alpha values
            alpha_bar_t = self.alphas_cumprod[t_val]
            alpha_bar_t_prev = self.alphas_cumprod[t_prev_val] if t_prev_val >= 0 else torch.ones(1, device=self.device)
            
            # Predict noise
            eps = model(x, t)
            
            # Predict x_0 from x_t and eps (the "x0-prediction" form of DDIM)
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
            
            if t_prev_val < 0:
                # Final step: return x0 prediction directly
                x = x0_pred
                break
            
            # DDIM variance parameter (sigma)
            # sigma = eta * sqrt((1 - alpha_t-1)/(1 - alpha_t)) * sqrt(1 - alpha_t/alpha_t-1)
            sigma = eta * ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)).sqrt() * \
                    (1 - alpha_bar_t / alpha_bar_t_prev).sqrt()
            
            # Direction pointing to x_t (the "predicted direction")
            dir_xt = (1 - alpha_bar_t_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            
            # Random noise (zero when eta=0)
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            
            # DDIM update step
            x = alpha_bar_t_prev.sqrt() * x0_pred + dir_xt + sigma * noise
        
        return x
