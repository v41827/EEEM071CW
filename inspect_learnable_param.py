import torch

# Path to your saved model weights
model_path = "logs/mobilenet_v3_small-veri/best_model.pth"

# Load the state dict from file
state_dict = torch.load(model_path, map_location="cpu")

print(f"✅ Loaded model from: {model_path}\n")
print("🔍 Parameters preview:")

# Print names, shapes, and first few values of each parameter
for name, param in state_dict.items():
    print(f"\n{name}: {param.shape}")
    print(param.flatten()[:10])  # Flatten and show first 10 values inside the tensor   