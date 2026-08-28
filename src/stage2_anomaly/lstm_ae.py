"""PyTorch LSTM Encoder-Decoder, real seq_len sequences (not the old seq_len=1
simplification). Single model with an .encode() method instead of Keras'
separate encoder sub-model -- same weights, one artifact to save/load.
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        return self.to_latent(h_n[-1])

    def forward(self, x):
        z = self.encode(x)
        h0 = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(h0)
        recon = self.output_layer(dec_out)
        return recon, z


def predict_recon_latent(model: LSTMAutoencoder, seqs: np.ndarray):
    """PyTorch equivalent of Keras' autoencoder.predict()/encoder.predict()
    -- returns (reconstruction, latent) as numpy arrays."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(seqs, dtype=torch.float32)
        recon, z = model(x)
    return recon.numpy(), z.numpy()


def train_lstm_ae(seqs: np.ndarray, seq_len: int, hidden_dim: int, latent_dim: int,
                   epochs: int, batch_size: int, lr: float = 1e-3,
                   val_split: float = 0.1, patience: int = 5):
    """Mimics Keras' validation_split + EarlyStopping(restore_best_weights=True)."""
    if len(seqs) == 0:
        raise ValueError(
            "No benign sequences of length seq_len could be built -- check "
            "there are enough benign rows in the raw Benign CSV."
        )

    n = len(seqs)
    n_val = max(int(n * val_split), 1)
    idx = np.random.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.tensor(seqs[train_idx], dtype=torch.float32)
    X_val = torch.tensor(seqs[val_idx], dtype=torch.float32)

    model = LSTMAutoencoder(seqs.shape[2], seq_len, hidden_dim, latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state, patience_ctr = float("inf"), None, 0
    loss_history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            batch = X_train[perm[i:i + batch_size]]
            opt.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        train_loss = total_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_recon, _ = model(X_val)
            val_loss = loss_fn(val_recon, X_val).item()

        print(f"  epoch {epoch+1}/{epochs}  loss={train_loss:.5f}  val_loss={val_loss:.5f}")
        loss_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  early stopping at epoch {epoch+1} (best val_loss={best_val:.5f})")
                break

    model.load_state_dict(best_state)

    recon_all, z_all = predict_recon_latent(model, seqs)
    centroid = z_all.mean(axis=0)
    recon_errors = np.mean(np.square(seqs - recon_all), axis=(1, 2))

    return model, centroid, loss_history, recon_errors
