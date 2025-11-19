import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class Neuron(nn.Module):
    def __init__(self, in_dim=784, dentritic_branch=28, synapse_per_branch=28, learning_rate=0.01):
        super().__init__()

        self.in_dim = in_dim
        self.dentritic_branch = dentritic_branch
        self.synapse_per_branch = synapse_per_branch
        self.lr = learning_rate
        self.dentrites = torch.stack([torch.exp(torch.randn(synapse_per_branch, device='cuda') * 0.9 - 2.0) for _ in range(dentritic_branch)], dim=0)
    
    def dentritic_branch_propagate(self, basal_features, learn=True, correct_label=None):
        basal_features = (basal_features > 0.0).float()
        basal_features = basal_features.view(self.dentritic_branch, self.synapse_per_branch)

        dentrites_activation = []
        for branch in range(self.dentritic_branch):
            basal_input = basal_features[branch]
            dentritic_synapse = self.dentrites[branch]
            activation = torch.tanh(basal_input @ dentritic_synapse)
            branch_fired = activation > 0.7

            if learn:
                if branch_fired:
                    if correct_label:
                        dentritic_synapse.data += 0.001 * basal_input
                    else:
                        dentritic_synapse.data -= 0.001 * basal_input
                else:
                    if correct_label:
                        dentritic_synapse.data += 0.001 * basal_input * 0.1
                continue

            dentrites_activation.append(activation)
        
        neuron_activation = None if len(dentrites_activation) == 0 else torch.tanh(torch.stack(dentrites_activation).sum())
        return neuron_activation
        
class PyramidalNeurons(nn.Module):
    def __init__(self, num_neurons=10):
        super().__init__()
        self.num_neurons = num_neurons
        self.neurons = [Neuron() for _ in range(num_neurons)]

    def print_neurons_synapses(self):
        for i, neuron in enumerate(self.neurons):
            synapses = neuron.dentrites.cpu().numpy().astype('uint8')
            synapses_img = synapses * 255
            img = Image.fromarray(synapses_img)
            img.save(f"./neurons-synapses/neuron {i}.png")

    def training_phase(self, image, label):
        image = image.flatten()

        for idx in range(self.num_neurons):
            if idx == label.item():
                self.neurons[idx].dentritic_branch_propagate(image, correct_label=True)
            else:
                self.neurons[idx].dentritic_branch_propagate(image, correct_label=False)

    def inference_phase(self, image):
        image = image.flatten()
        neurons_firing = []
        for neuron in self.neurons:
            firing_rate = neuron.dentritic_branch_propagate(image, learn=False)
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
# model = PyramidalNeurons()
# print(model.training_phase(x, y))
# print(model.inference_phase(x))