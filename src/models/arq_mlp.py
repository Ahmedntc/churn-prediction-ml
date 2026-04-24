import torch
import torch.nn as nn

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

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)