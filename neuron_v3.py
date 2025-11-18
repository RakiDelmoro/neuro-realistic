import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class PyramidalNeuron(nn.Module):
    def __init__(self, in_dim=784, dentritic_branch=28, branch_synapses=28):
        super().__init__()

        self.in_dim = in_dim
        self.branches = dentritic_branch
        self.branch_size = branch_synapses
        self.branches_synapses = torch.zeros(dentritic_branch, branch_synapses, device='cuda')
    
    def plasticity(self, sensory_input):
        sensory_input = (sensory_input > 0.5).float()
        sensory_input = sensory_input.view(self.branches, self.branch_size)

        for branch in range(self.branches):
            branch_synapses = self.branches_synapses[branch]
            active_sensory_input = torch.where(sensory_input[branch] > 0)[0]

            branch_synapses[active_sensory_input] = 1

    def propagate_to_soma(self, basal_features):
        basal_features = basal_features.view(self.branches, self.branch_size)
        soma_rate = 0
        for branch in range(self.branches):
            feature = torch.where(basal_features[branch] > 0)[0]
            synapse = torch.where(self.branches_synapses[branch] > 0)[0]
            correctness = len(synapse) - len(feature)
            soma_rate += correctness

        return soma_rate
        
class Neurons(nn.Module):
    def __init__(self, num_neurons=10):
        super().__init__()
        self.neurons = [PyramidalNeuron() for _ in range(num_neurons)]
        # Track which digits each neuron has been trained on
        self.trained_digits = [set() for _ in range(num_neurons)]
        
    def training_phase(self, image, label):
        """
        Returns True if training should continue, False if all neurons have seen all digits
        """
        image = image.flatten()
        label_item = label.item()
        
        neuron = self.neurons[label_item]
        
        # Only train if this neuron hasn't seen this digit yet
        if label_item not in self.trained_digits[label_item]:
            neuron.plasticity(image)
            self.trained_digits[label_item].add(label_item)
        
        # Check if all neurons have been trained on all digits
        return not self.is_training_complete()
    
    def is_training_complete(self):
        """
        Check if each neuron has been trained on all digits 0-9
        """
        for digit_set in self.trained_digits:
            if len(digit_set) < 10:
                return False
        return True

    def print_neurons_synapses(self):
        for i, neuron in enumerate(self.neurons):
            synapses = neuron.branches_synapses.cpu().numpy().astype('uint8')
            synapses_img = synapses * 255
            img = Image.fromarray(synapses_img)
            img.save(f"./neurons-synapses/neuron {i}.png")

    def inference_phase(self, image):
        basal_features = (image > 0.5).float()

        neurons_firing = []
        for neuron in self.neurons:
            synapses = neuron.branches_synapses.flatten()
            firing_rate = (basal_features * synapses).sum()
            neurons_firing.append(firing_rate.item())
        
        return torch.tensor(neurons_firing).argmax()
    
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

# x = torch.randn(784, device='cuda')
# y = torch.randint(0, 10, size=(1,))
# model = Neurons()
# print(model.training_phase(x, y))
# print(model.inference_phase(x))