"""
LSTM Training Script (Abdul Rehman)
====================================
Trains the dual-head ExerciseLSTM with early stopping.

TRAINING CONCEPTS (for viva):
  - Loss = MSE(rep_count) + λ * BCE(form_quality)
  - MSE (Mean Squared Error): for regression tasks (rep counting)
  - BCE (Binary Cross-Entropy): for classification tasks (form quality)
  - Early stopping: stop training when validation loss stops improving
  - This prevents overfitting (memorizing training data instead of learning patterns)

Usage:
  python -m lstm_module.train --data_dir data/processed --epochs 100
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime

from lstm_module.model import ExerciseLSTM
from lstm_module.dataset import build_datasets


def train_model(data_dir, output_dir='models', epochs=100, batch_size=32,
                lr=0.001, lambda_form=0.5, patience=10, window_size=60,
                resume=False, checkpoint_every=5):
    """
    Train the ExerciseLSTM model.

    Args:
        data_dir: path to data/processed/
        output_dir: where to save model weights
        epochs: max training epochs
        batch_size: samples per batch
        lr: learning rate for Adam optimizer
        lambda_form: weight for form loss (total = rep_loss + λ * form_loss)
        patience: early stopping patience (epochs without improvement)
        window_size: sliding window size
        resume: if True, resume from last checkpoint (for Colab disconnects)
        checkpoint_every: save checkpoint every N epochs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build datasets
    print("Building datasets...")
    train_ds, val_ds, test_ds = build_datasets(
        data_dir, window_size=window_size
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ExerciseLSTM(input_size=84).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions
    rep_criterion = nn.MSELoss()        # For rep counting (regression)
    form_criterion = nn.BCELoss()       # For form quality (classification)

    # Optimizer: Adam with learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [],
               'train_rep_loss': [], 'train_form_loss': [],
               'val_rep_loss': [], 'val_form_loss': []}

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # ===== RESUME FROM CHECKPOINT =====
    checkpoint_path = output_path / 'checkpoint.pth'
    if resume and checkpoint_path.exists():
        print("\n🔄 Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        history = checkpoint['history']
        print(f"   Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    elif resume:
        print("⚠️ --resume flag set but no checkpoint found. Starting fresh.")

    print(f"\nStarting training for epochs {start_epoch+1}-{epochs}...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Rep Loss':>10} | {'Form Loss':>10}")
    print("-" * 60)

    for epoch in range(start_epoch, epochs):
        # ===== TRAINING PHASE =====
        model.train()
        train_losses = {'total': [], 'rep': [], 'form': []}

        for x_batch, rep_true, form_true in train_loader:
            x_batch = x_batch.to(device)
            rep_true = rep_true.to(device)
            form_true = form_true.to(device)

            # Forward pass
            rep_pred, form_pred = model(x_batch)

            # Compute combined loss
            rep_loss = rep_criterion(rep_pred, rep_true)
            form_loss = form_criterion(form_pred, form_true)
            total_loss = rep_loss + lambda_form * form_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_losses['total'].append(total_loss.item())
            train_losses['rep'].append(rep_loss.item())
            train_losses['form'].append(form_loss.item())

        # ===== VALIDATION PHASE =====
        model.eval()
        val_losses = {'total': [], 'rep': [], 'form': []}

        with torch.no_grad():
            for x_batch, rep_true, form_true in val_loader:
                x_batch = x_batch.to(device)
                rep_true = rep_true.to(device)
                form_true = form_true.to(device)

                rep_pred, form_pred = model(x_batch)
                rep_loss = rep_criterion(rep_pred, rep_true)
                form_loss = form_criterion(form_pred, form_true)
                total_loss = rep_loss + lambda_form * form_loss

                val_losses['total'].append(total_loss.item())
                val_losses['rep'].append(rep_loss.item())
                val_losses['form'].append(form_loss.item())

        # Average losses
        avg_train = np.mean(train_losses['total'])
        avg_val = np.mean(val_losses['total'])

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['train_rep_loss'].append(np.mean(train_losses['rep']))
        history['train_form_loss'].append(np.mean(train_losses['form']))
        history['val_rep_loss'].append(np.mean(val_losses['rep']))
        history['val_form_loss'].append(np.mean(val_losses['form']))

        # Print progress
        print(f"{epoch+1:>5} | {avg_train:>10.4f} | {avg_val:>10.4f} | "
              f"{np.mean(val_losses['rep']):>10.4f} | {np.mean(val_losses['form']):>10.4f}")

        # ===== EARLY STOPPING =====
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), output_path / 'lstm_best.pth')
            print(f"       → Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        # ===== PERIODIC CHECKPOINT (for Colab T4 disconnects) =====
        if (epoch + 1) % checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'history': history,
            }, output_path / 'checkpoint.pth')
            print(f"       💾 Checkpoint saved at epoch {epoch+1}")

    # ===== FINAL EVALUATION ON TEST SET =====
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    model.load_state_dict(torch.load(output_path / 'lstm_best.pth', weights_only=True))
    model.eval()

    all_rep_true, all_rep_pred = [], []
    all_form_true, all_form_pred = [], []

    with torch.no_grad():
        for x_batch, rep_true, form_true in test_loader:
            x_batch = x_batch.to(device)
            rep_pred, form_pred = model(x_batch)

            all_rep_true.extend(rep_true.numpy().flatten())
            all_rep_pred.extend(rep_pred.cpu().numpy().flatten())
            all_form_true.extend(form_true.numpy().flatten())
            all_form_pred.extend(form_pred.cpu().numpy().flatten())

    # Rep counting metrics
    rep_true = np.array(all_rep_true)
    rep_pred = np.round(np.array(all_rep_pred))  # Round to nearest integer
    rep_mae = np.mean(np.abs(rep_true - rep_pred))
    rep_accuracy = np.mean(rep_true == rep_pred)

    # Form classification metrics
    form_true = np.array(all_form_true)
    form_pred = (np.array(all_form_pred) > 0.5).astype(float)
    form_accuracy = np.mean((form_true > 0.5) == (form_pred > 0.5))

    print(f"Rep Count MAE:       {rep_mae:.2f}")
    print(f"Rep Count Accuracy:  {rep_accuracy:.1%}")
    print(f"Form Classification: {form_accuracy:.1%}")

    # Save training history and results
    results = {
        'history': history,
        'test_results': {
            'rep_mae': float(rep_mae),
            'rep_accuracy': float(rep_accuracy),
            'form_accuracy': float(form_accuracy),
        },
        'config': {
            'epochs_trained': epoch + 1,
            'batch_size': batch_size,
            'lr': lr,
            'lambda_form': lambda_form,
            'window_size': window_size,
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_path / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Training complete! Model saved to {output_path / 'lstm_best.pth'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ExerciseLSTM')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
