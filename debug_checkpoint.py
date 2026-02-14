import torch
import numpy as np

# Load and inspect the checkpoint
checkpoint = torch.load('./hierarchical_checkpoints/final.pt', map_location='cpu')

print("Checkpoint contents:")
for agent_id, weights in checkpoint.items():
    print(f"\n{agent_id}:")
    if hasattr(weights, 'keys'):
        for key, tensor in weights.items():
            print(f"  {key}: shape={tensor.shape}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")
    else:
        print(f"  Type: {type(weights)}")

# Test if models can produce actions
from examples.train_ppo_roles_3v3_simple import SimplePolicy

# Create a dummy observation
obs_dim = 64
action_dim = 17
dummy_obs = np.random.randn(obs_dim).astype(np.float32)

print(f"\nTesting with dummy observation: {dummy_obs[:5]}...")

for agent_id, weights in checkpoint.items():
    if hasattr(weights, 'keys'):  # It's a state dict
        model = SimplePolicy(obs_dim, action_dim)
        model.load_state_dict(weights)
        model.eval()
        
        with torch.no_grad():
            state = torch.FloatTensor(dummy_obs)
            logits = model(state)
            probs = torch.softmax(logits, dim=-1)
            
        print(f"\n{agent_id} model output:")
        print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        print(f"  Probs range: [{probs.min().item():.6f}, {probs.max().item():.6f}]")
        print(f"  Most likely action: {probs.argmax().item()}")
