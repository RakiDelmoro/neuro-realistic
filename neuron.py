import torch
import torch.nn as nn

class PyramidalNeuron(nn.Module):
    def __init__(self, basal_size, image_size=784, num_classes=10, sparsity=0.03):
        super().__init__()

        self.basal_size = basal_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.sparsity = sparsity

        self.basal_synapses = torch.zeros(num_classes, basal_size, device='cuda')    
        self.input_projection = torch.randn(image_size, basal_size, device='cuda') * 0.1

    def basal_encoder(self, image):
        # Project to basal size
        projected = torch.matmul(image, self.input_projection)
        # Sparse encoding: keep top 3% activations
        k = int(round(self.basal_size * self.sparsity))
        _, top_indices = torch.topk(projected, k)
        basal_features = torch.zeros_like(projected, device='cuda')
        basal_features.scatter_(1, top_indices, 1.0)
        return basal_features

    def training_phase(self, image, label):
        basal_features = self.basal_encoder(image)
        # Accumulate with learning rate
        self.basal_synapses[label] += 0.01 * basal_features
        self.basal_synapses[label] = torch.clamp(self.basal_synapses[label], 0, 1)

    def inference_phase(self, image):
        basal_features = self.basal_encoder(image).unsqueeze(0)
        neurons_firing_rate = torch.matmul(basal_features, self.basal_synapses.t())

        return torch.argmax(neurons_firing_rate).item()

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
