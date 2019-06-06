import torch
from typing import Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")