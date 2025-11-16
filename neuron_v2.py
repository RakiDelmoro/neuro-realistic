import torch
import torch.nn as nn

class PyramidalNeuron(nn.Module):
    def __init__(self, basal_size, image_size=784, num_classes=10, sparsity=0.03):
        super().__init__()

        self.basal_size = basal_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.sparsity = sparsity

        self.basal_synapses =  torch.rand(num_classes, basal_size, device='cuda') * 0.01
        self.threshold = 0.5 * self.sparsity * self.basal_size

        # Add learning rate parameters
        self.lr_potentiation = 0.01  # For target class
        self.lr_depression = 0.001   # For non-target classes

    def basal_encoder(self, image):
        basal_features = torch.zeros(1, self.basal_size, device=image.device)
        
        # Get active pixels from image
        active_pixels = torch.where(image > 0.5)[1]
        
        # Randomly map to basal dendrites (with modulo to stay in bounds)
        if len(active_pixels) > 0:
            basal_indices = active_pixels % self.basal_size  # Wrap around
            basal_features[:, basal_indices] = 1.0
        
        return basal_features

    def plasticity(self, basal_features, label):
        basal_flat = basal_features.flatten()
        
        # Potentiation for correct class (Hebbian learning)
        self.basal_synapses[label] += self.lr_potentiation * basal_flat
        
        # Gentler depression for incorrect classes
        for each in range(self.num_classes):
            if each != label:
                # Only depress where there's overlap
                overlap = self.basal_synapses[each] * basal_flat
                self.basal_synapses[each] -= self.lr_depression * overlap
        
        # Normalize to prevent unbounded growth
        self.basal_synapses.clamp_(0.0, 1.0)

    def training_phase(self, image, label):
        basal_features = self.basal_encoder(image)
        self.plasticity(basal_features, label)

    def inference_phase(self, image):
        basal_features = self.basal_encoder(image)
        
        overlaps = (self.basal_synapses * basal_features).sum(dim=1)
        predicted_label = torch.argmax(overlaps)
        
        return predicted_label

    def runner(self, train_loader=None, test_loader=None, train=True):

        if train and train_loader is not None:
            for train_image, train_label in train_loader:
                image = torch.tensor(train_image, device='cuda')
                label = torch.tensor(train_label, device='cuda')
                self.training_phase(image, label)

        if not train and test_loader is not None:       
            correctness = []
            for test_image, test_label in test_loader:
                test_image = torch.tensor(test_image, device='cuda')
                test_label = torch.tensor(test_label, device='cuda')
                predicted = self.inference_phase(test_image)
                correctness.append(predicted == test_label.item())

            if sum(correctness) == 0:
                return 0.0
            return sum(correctness) / len(correctness)
        
# x = torch.randn(1, 784, device='cuda')
# y = torch.randint(0, 10, (1,), device='cuda')
# neuron = PyramidalNeuron(basal_size=1024)
# print(neuron.training_phase(x, y))
# print(neuron.inference_phase(x))
