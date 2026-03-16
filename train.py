from dit import DiT
from comet_ml import Experiment
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict

from model import MeanFlow
from fid_evaluation import FIDEvaluation
import os
os.environ["COMET_API_KEY"] = "0tXk5XHql4WqhBohZ2EI8RqX6"


import os
import torch.optim as optim
import torch.nn.functional as F
from accelerate import Accelerator


def main():
    n_steps = 500000  # Reduced for MNIST (simpler dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128  # Can use larger batch size for MNIST
    
    # MNIST image settings
    image_size = 28  # MNIST native size (or use 32 for power of 2)
    image_channels = 1  # MNIST is grayscale
    num_classes = 10  # MNIST has 10 digit classes (0-9)
    
    # Create directories
    os.makedirs('./images_mnist', exist_ok=True)
    os.makedirs('./results_mnist', exist_ok=True)
    checkpoint_root_path = './checkpoint/dit_mnist/'
    os.makedirs(checkpoint_root_path, exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')

    # Initialize Comet ML experiment
    experiment = Experiment(
        project_name="meanflow",
    )
    
    # Log hyperparameters
    experiment.log_parameters({
        "dataset": "MNIST",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "model": "DiT-MNIST",
        "dim": 256,  # Smaller model for MNIST
        "depth": 8,   # Fewer layers needed
        "num_heads": 4,  # Fewer attention heads
        "patch_size": 4,  # Larger patches for MNIST (28/4 = 7x7 patches)
        "dropout_prob": 0.1,
        "optimizer": "Adam",
        "mixed_precision": "bfloat16",
        "fid_subset_size": 1000,
        "image_size": image_size,
        "image_channels": image_channels,
        "training_cfg_rate": 0.2,
        "lambda_weight": 0.05,
        "sigma_min": 1e-06,
    })

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])
    
    dataset = torchvision.datasets.MNIST(
        root="./",
        train=True,
        download=True,
        transform=transform,
    )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=8  # Reduced from 40
    )
    train_dataloader = cycle(train_dataloader)

    # Create model for MNIST generation
    model = DiT(
        input_size=image_size,
        patch_size=4,  # Larger patch size for MNIST
        in_channels=image_channels,
        dim=256,  # Smaller dimension
        depth=8,  # Fewer layers
        num_heads=4,  # Fewer heads
        num_classes=num_classes,
        learn_sigma=False,
        class_dropout_prob=0.1,
    ).to(accelerator.device)

    ema_model = deepcopy(model).eval()
    sampler = MeanFlow(
        device=accelerator.device,
        channels=image_channels,
        image_size=image_size,
        num_classes=num_classes,
        cfg_drop_prob=0.1
    )
    ema_decay = 0.9999
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # FID evaluation setup for MNIST
    fid_subset_size = 1000
    
    def limited_cycle(iterable, max_batches):
        batch_count = 0
        while batch_count < max_batches:
            for i in iterable:
                yield i
                batch_count += 1
                if batch_count >= max_batches:
                    break
    
    fid_batches_needed = (fid_subset_size + batch_size - 1) // batch_size
    
    fid_dataset = torchvision.datasets.MNIST(
        root="./",
        train=True,
        download=False,
        transform=transform,
    )
    
    fid_dataloader = torch.utils.data.DataLoader(
        fid_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False, 
        num_workers=4
    )
    
    fid_dataloader_limited = limited_cycle(fid_dataloader, fid_batches_needed)
    fid_eval = FIDEvaluation(batch_size, fid_dataloader_limited, sampler, num_fid_samples=100)
    
    def update_ema(ema_model, model, decay):
        """Update EMA model parameters"""
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.lerp_(p.data, 1 - decay)
    
    def sample_and_log_images():
        cfg_scale = 5.0
        ema_model.eval()
        with torch.no_grad():
            # Use the model's own sampling method
            # 多步采样（通常是20-50步），追求高质量
            samples = sampler.sample_each_class(ema_model, 10)
            # 一步采样（关键！），验证 Mean Flow 的拉直效果
            sampels_one_step = sampler.sample_each_class(ema_model, 10, sample_steps=1)
            
            # For grayscale images, we need to handle the channel dimension
            if image_channels == 1:
                # Convert grayscale to RGB for visualization
                samples_rgb = samples.repeat(1, 3, 1, 1)
                samples_one_step_rgb = sampels_one_step.repeat(1, 3, 1, 1)
                log_img = make_grid(samples_rgb, nrow=10, normalize=True, value_range=(-1, 1))
                log_img_one_step = make_grid(samples_one_step_rgb, nrow=10, normalize=True, value_range=(-1, 1))
            else:
                log_img = make_grid(samples, nrow=10, normalize=True, value_range=(-1, 1))
                log_img_one_step = make_grid(sampels_one_step, nrow=10, normalize=True, value_range=(-1, 1))
            
            img_save_path = f"./images_mnist/step{step}_cfg{cfg_scale}.png"
            save_image(log_img, img_save_path)
            img_save_path_one_step = f"./images_mnist/step{step}_one_step.png"
            save_image(log_img_one_step, img_save_path_one_step)
            experiment.log_image(
                img_save_path,
                name=f"cfg_{cfg_scale}",
                step=step
            )
            experiment.log_image(
                img_save_path_one_step,
                name=f"one_step",
                step=step
            )

    losses = []
    mse_vals = []
    sigma_min = 1e-06
    training_cfg_rate = 0.1
    lambda_weight = 0.001
    use_immiscible = True
    gradient_clip = 1.0
    model.train()
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training DiT on MNIST")
        for step in pbar:
            data = next(train_dataloader)
            optimizer.zero_grad(set_to_none=True)
            
            # Get images and labels
            x1 = data[0].to(accelerator.device)
            y = data[1].to(accelerator.device)
            b = x1.shape[0]


            if use_immiscible:
                k = 8
                # Generate k noise samples for each data point
                z_candidates = torch.randn(b, k, image_channels, image_size, image_size, device=x1.device, dtype=x1.dtype)
                
                x1_flat = x1.flatten(start_dim=1)  # [b, c*h*w]
                z_candidates_flat = z_candidates.flatten(start_dim=2)  # [b, k, c*h*w]
                
                # Compute distances between each data point and its k noise candidates

                distances = torch.norm(x1_flat.unsqueeze(1) - z_candidates_flat, dim=2)  # [b, k]
                
                # Find the farthest noise sample for each data point (for immiscible)
                _, min_indices = torch.min(distances, dim=1)  # [b]
                
                # Method 1: Using gather with proper indexing
                batch_indices = torch.arange(b, device=x1.device)
                z = z_candidates[batch_indices, min_indices]  # [b, c, h, w]
                
            else:
                # Standard noise sampling
                z = torch.randn_like(x1)
                

            
            loss, current_lambda = sampler.loss(model, x1, z, y, step=step)
                
            accelerator.backward(loss)

            # Calculate and clip gradient norm
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

            optimizer.step()
            
            # Update EMA
            update_ema(ema_model, model, ema_decay)

            # Logging
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item(), "lambda": current_lambda})
            # pbar.set_postfix({"loss": loss.item(),  "mse_val": mse_val.item()})

            experiment.log_metric("loss", loss.item(), step=step)
            experiment.log_metric("grad_norm", grad_norm.item(), step=step)
            experiment.log_metric("lambda", current_lambda, step=step)
            experiment.log_metric("lr", optimizer.param_groups[0]['lr'], step=step)

            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                experiment.log_metric("avg_loss_100", avg_loss, step=step)
                
            if step % 1000 == 0 or step == n_steps - 1:  # More frequent for MNIST
                avg_loss = sum(losses) / len(losses) if losses else 0
                print(f"\nStep: {step+1}/{n_steps} | avg_loss: {avg_loss:.4f}")
                losses.clear()
                
                sample_and_log_images()
                model.train()

            if step % 2500 == 0 or step == n_steps - 1:  # More frequent for MNIST
                # FID evaluation
                try:
                    print(f"Running FID evaluation on {fid_subset_size} samples...")
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        fid_score = fid_eval.fid_score(ema_model)
                    print(f"FID score at step {step}: {fid_score}")
                    experiment.log_metric("FID", fid_score, step=step)
                except Exception as e:
                    print(f"FID evaluation failed: {e}")
                model.train()
                
                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_root_path, f"step_{step}.pth")
                state_dict = {
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": {
                        "dataset": "MNIST",
                        "image_size": image_size,
                        "image_channels": image_channels,
                        "sigma_min": sigma_min,
                        "lambda_weight": lambda_weight,
                        "training_cfg_rate": training_cfg_rate,
                        "gradient_clip": gradient_clip,
                        "ema_decay": ema_decay,
                    }
                }
                torch.save(state_dict, checkpoint_path)
                
                experiment.log_model(
                    name=f"checkpoint_step_{step}",
                    file_or_folder=checkpoint_path
                )

    # Final save
    checkpoint_path = os.path.join(checkpoint_root_path, "model_mnist_final.pth")
    state_dict = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "config": {
            "dataset": "MNIST",
            "image_size": image_size,
            "image_channels": image_channels,
            "sigma_min": sigma_min,
            "lambda_weight": lambda_weight,
            "training_cfg_rate": training_cfg_rate,
            "gradient_clip": gradient_clip,
            "ema_decay": ema_decay,
        }
    }
    torch.save(state_dict, checkpoint_path)
    
    experiment.log_model(
        name="final_model",
        file_or_folder=checkpoint_path
    )
    
    experiment.end()


if __name__ == "__main__":
    main()
