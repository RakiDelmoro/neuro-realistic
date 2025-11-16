import torch
import torch.nn as nn

class PyramidalNeuron(nn.Module):
    def __init__(self, basal_size, image_size=784, num_classes=10, sparsity=0.03):
        super().__init__()

        self.basal_size = basal_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.sparsity = sparsity

        # Initialize synapses 0.3 to 0.7
        self.basal_synapses =  torch.zeros(num_classes, basal_size, device='cuda')

    def basal_encoder(self, image):
        basal_features = torch.zeros(1, self.basal_size, device=image.device)
        active_indices = torch.where(image > 0.7)[1]
        basal_features[:, active_indices] = 1.0

        return basal_features

    def plasticity(self, basal_features, label):
        # Union sdr Bitwise OR to label and exclude to other union
        for each in range(self.num_classes):
            if each == label:
                self.basal_synapses[each] = (self.basal_synapses[label].bool() | basal_features.bool()).float()
            # else:
            #     # Exclude basal_features from other unions using AND NOT
            #     self.basal_synapses[each] = (self.basal_synapses[each].bool() & ~basal_features.bool()).float()

    def training_phase(self, image, label):
        basal_features = self.basal_encoder(image)
        self.plasticity(basal_features, label)

    def inference_phase(self, image):
        basal_features = self.basal_encoder(image)
        
        # Calculate overlap with each union
        overlaps = torch.zeros(self.num_classes)
        for each in range(self.num_classes):
            # Count how many active bits match (intersection)
            overlap = (basal_features.bool() & self.basal_synapses[each].bool()).sum()
            overlaps[each] = overlap
        
        # Predict the class with maximum overlap
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
