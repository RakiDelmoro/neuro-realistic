import torch

class SupervisedHTMNeuronTorch:
    def __init__(self, image_size, label_size, proximal_threshold=2, distal_threshold=2, device='cpu'):
        self.image_size = image_size
        self.label_size = label_size
        self.proximal_threshold = proximal_threshold
        self.distal_threshold = distal_threshold
        self.device = device

        # Binary synapses initialized randomly
        self.proximal_synapses = torch.randint(0, 2, (image_size,), dtype=torch.float32, device=device)
        self.distal_synapses = torch.randint(0, 2, (label_size,), dtype=torch.float32, device=device)

    def train(self, image_sdr, label_sdr):
        # Convert to torch tensors if needed
        if not isinstance(image_sdr, torch.Tensor):
            image_sdr = torch.tensor(image_sdr, dtype=torch.float32, device=self.device)
        if not isinstance(label_sdr, torch.Tensor):
            label_sdr = torch.tensor(label_sdr, dtype=torch.float32, device=self.device)

        # Hebbian-like strengthening
        self.proximal_synapses = torch.where(image_sdr == 1, torch.ones_like(self.proximal_synapses), self.proximal_synapses)
        self.distal_synapses = torch.where(label_sdr == 1, torch.ones_like(self.distal_synapses), self.distal_synapses)

    def activate(self, image_sdr, label_sdr):
        # Convert to torch tensors if needed
        if not isinstance(image_sdr, torch.Tensor):
            image_sdr = torch.tensor(image_sdr, dtype=torch.float32, device=self.device)
        if not isinstance(label_sdr, torch.Tensor):
            label_sdr = torch.tensor(label_sdr, dtype=torch.float32, device=self.device)

        # Calculate activations
        proximal_active = torch.sum(image_sdr * self.proximal_synapses)
        distal_active = torch.sum(label_sdr * self.distal_synapses)

        # Fire only if both thresholds are met
        fired = (proximal_active >= self.proximal_threshold) and (distal_active >= self.distal_threshold)
        return fired

# -----------------------------
# Example usage

image_sdr = torch.tensor([1,0,1,0,1,0,0,1,0,0], dtype=torch.float32)
label_sdr = torch.tensor([0,0,1,0,0], dtype=torch.float32)

neuron = SupervisedHTMNeuronTorch(image_size=10, label_size=5, proximal_threshold=3, distal_threshold=1)

# Train neuron
neuron.train(image_sdr, label_sdr)

# Correct pair
print("Correct pair fires:", neuron.activate(image_sdr, label_sdr))

# Incorrect label
wrong_label_sdr = torch.tensor([1,0,0,0,0], dtype=torch.float32)
print("Incorrect pair fires:", neuron.activate(image_sdr, wrong_label_sdr))
