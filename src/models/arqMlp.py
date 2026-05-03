import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class ChurnMLP(nn.Module):
    def __init__(
        self,
        input_dim: int, #numero de features no nosso caso 46
        hidden_dims: list[int] = [64, 32, 16], #camadas ocultas com 64, 32 e 16 neurônios respectivamente
        dropout_rate: float = 0.3, #taxa de dropout para evitar overfitting

    ) -> None:
        super().__init__()


        layers: list[nn.Module] = []
        in_features = input_dim

        for out_features in hidden_dims:
            layers += [
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ]
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)

    def prever_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class ChurnMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper sklearn para o ChurnMLP — permite usar o modelo PyTorch
    dentro de um Pipeline sklearn.
    """
    def __init__(
        self,
        hidden_dims: list[int] = [64, 32, 16],
        dropout_rate: float = 0.3,
        lr: float = 0.01,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 12,
    ):
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.model_ = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.array(x)  # garante numpy
        y = np.array(y)  # converte Series para numpy
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

        x_tr, x_val, y_tr, y_val = train_test_split(
            x, y, test_size=0.15, random_state=self.random_state, stratify=y
        )

        # pos_weight para lidar com desbalanceamento
        non_churners = (y_tr == 0).sum()
        churners = (y_tr == 1).sum()
        pos_weight = torch.tensor([non_churners / churners], dtype=torch.float32).to(self.device_)

        x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        x_val_t = torch.tensor(x_val, dtype=torch.float32).to(self.device_)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device_)

        from torch.utils.data import DataLoader, TensorDataset
        train_loader = DataLoader(
            TensorDataset(x_tr_t, y_tr_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.model_ = ChurnMLP(
            input_dim=x_tr.shape[1],
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
        ).to(self.device_)
 # BCEWithLogitsLoss é uma combinação de sigmoid + binary cross-entropy, e aceita para pos_weight ajuda a lidar com o desbalanceamento que temos no dataset
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        epoch = 1
        while epoch <= self.max_epochs and patience_counter < self.patience:
            self.model_.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device_), y_batch.to(self.device_)
                optimizer.zero_grad()
                loss = criterion(self.model_(x_batch), y_batch)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(x_val_t), y_val_t).item()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            epoch += 1

        logger.info("Treino encerrado na época %d", epoch - 1)
        self.model_.load_state_dict(best_state)
        return self

    # predict_proba retorna a probabilidade de cada classe (churner e não churner), e predict retorna a classe prevista (0 ou 1)
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(x, dtype=torch.float32).to(self.device_)
        probs = self.model_.prever_proba(x_t).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)
