import torch
import torch.nn as nn

class BasalEncoder:
    def __init__(self, input_size, output_size, sparsity=0.03, seed=42):
        self.input_size = input_size
        self.output_size = output_size
        self.sparsity = sparsity
        self.num_active = int(round(output_size * sparsity))
        
        # Initialize random projection for encoding
        torch.manual_seed(seed)
        self.projection = torch.randn(input_size, output_size, device='cuda') * 0.1
    
    def encode(self, x):
        # Handle single sample vs batch
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        
        # Project to output space
        activations = torch.matmul(x, self.projection)
        
        # Get top-k indices for sparsification
        _, top_indices = torch.topk(activations, self.num_active, dim=1)
        
        # Create sparse representation
        sdr = torch.zeros_like(activations, device='cuda')
        sdr.scatter_(1, top_indices, 1.0)
        
        if squeeze:
            sdr = sdr.squeeze(0)
        
        return sdr

class ApicalEncoder:
    def __init__(self, num_categories, output_size, sparsity=0.03):
        self.encodings = torch.zeros(num_categories, output_size, device='cuda')
        for i in range(num_categories):
            active_bits = torch.randperm(output_size)[:int(output_size * sparsity)]
            self.encodings[i, active_bits] = 1.0

    def encode(self, label):
        return self.encodings[label]

class PyramidalNeuron(nn.Module):
    def __init__(self, basal_size, image_size=784, num_classes=10):
        super().__init__()

        self.basal_size = basal_size
        self.image_size = image_size
        self.num_classes = num_classes

        self.basal_encoder = BasalEncoder(input_size=image_size, output_size=basal_size, sparsity=0.03)
        self.basal_synapses = torch.zeros(num_classes, basal_size, device='cuda')
        
    def training_phase(self, image, label):
        basal_features = self.basal_encoder.encode(image)
        # Bit OR operation
        self.basal_synapses[label] = (self.basal_synapses[label].bool() | basal_features.flatten().bool()).float()

    def inference_phase(self, image):
        basal_features = self.basal_encoder.encode(image).flatten()
        basal_overlap = torch.mv(self.basal_synapses, basal_features)

        return torch.argmax(basal_overlap).item()

    def runner(self, train_loader, test_loader):
        for train_image, train_label in train_loader:
            image = torch.tensor(train_image, device='cuda')
            label = torch.tensor(train_label, device='cuda')
            self.training_phase(image, label)

        correctness = []
        for test_image, test_label in test_loader:
            test_image = torch.tensor(test_image, device='cuda')
            test_label = torch.tensor(test_label, device='cuda')
            predicted = self.inference_phase(test_image)
            correctness.append(predicted == test_label.item())

        return sum(correctness) / len(correctness)
