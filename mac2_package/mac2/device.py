"""Device selection and shared constants -- imported first by everything else."""
import torch

EPS = 1e-15

if not torch.cuda.is_available():
    device = torch.device('cpu')
    print("No GPU — using CPU")
else:
    _name = torch.cuda.get_device_name(0)
    _cap = torch.cuda.get_device_capability(0)
    if _cap[0] > 3 or (_cap[0] == 3 and _cap[1] >= 7):
        device = torch.device('cuda')
        print(f"GPU: {_name} (CUDA {_cap[0]}.{_cap[1]})")
    else:
        device = torch.device('cpu')
        print(f"GPU too old — using CPU ({_name})")
print(f"Active device: {device}")
