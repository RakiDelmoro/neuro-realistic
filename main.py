import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuron import PyramidalNeuron

def plot_training_progress(mlp_accuracies, agent_accuracies, filename='training_progress.png'):
    epochs = list(range(len(mlp_accuracies)))  # x-axis = epoch numbers

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, mlp_accuracies, label='MLP Accuracy', color='blue', marker='o')
    plt.plot(epochs, agent_accuracies, label='Agent Accuracy', color='orange', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Progress: Accuracy per Epoch')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot to file
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"✅ Training progress plot saved as '{filename}'")

def mnist_dataloader(img_arr, label_arr, batch_size, shuffle):
    num_samples = img_arr.shape[0]    
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size

        # img_arr = image_to_patches(img_arr, patch_width=14)
        batched_img = img_arr[indices[start_idx:end_idx]]
        batched_label = label_arr[indices[start_idx:end_idx]]

        yield batched_img, batched_label

def standard_mlp(model, train_loader=None, test_loader=None, train=True):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    if train and train_loader is not None:
        # Training
        for train_image, train_label in train_loader:
            image = torch.tensor(train_image, device='cuda')
            label = torch.tensor(train_label, device='cuda')

            model_pred = model(image)
            loss = loss_fn(model_pred, label)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if not train and test_loader is not None:
        correctness = []
        for test_image, test_label in test_loader:
            test_image = torch.tensor(test_image, device='cuda')
            test_label = torch.tensor(test_label, device='cuda')

            model_pred = model(test_image).argmax(dim=-1).item()
            correctness.append(model_pred == test_label.item())

        return sum(correctness) / len(correctness)


def training_runner():
    # Load datasets
    with open('./dataset/mnist-digits.pkl', 'rb') as f: ((train_digit_images, train_digit_labels), (test_digit_images, test_digit_labels), _) = pickle.load(f, encoding='latin1')
    with open('./dataset/mnist-fashion.pkl', 'rb') as f: ((train_fashion_images, train_fashion_labels), (test_fashion_images, test_fashion_labels), _) = pickle.load(f, encoding='latin1')

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(784, 1024, device='cuda')
            self.layer2 = torch.nn.Linear(1024, 10, device='cuda')
        def forward(self, input_x):
            return self.layer2(self.layer1(input_x).relu())

    mlp = Mlp()
    neuron = PyramidalNeuron(basal_size=10000)

    mlp_accuracies = []
    agent_accuracies = []
    for epoch in range(1, 101):
        # mnist-digit loaders
        digit_train_loader = mnist_dataloader(train_digit_images, train_digit_labels, batch_size=1, shuffle=True)
        digit_test_loader = mnist_dataloader(test_digit_images, test_digit_labels, batch_size=1, shuffle=False)
        # mnist-fashion loaders
        fashion_train_loader = mnist_dataloader(train_fashion_images, train_fashion_labels, batch_size=1, shuffle=True)
        fashion_test_loader = mnist_dataloader(test_fashion_images, test_fashion_labels, batch_size=1, shuffle=False)
        # # MLP train and test
        standard_mlp(model=mlp, train_loader=digit_train_loader, train=True)
        standard_mlp(model=mlp, train_loader=fashion_train_loader, train=True)
        digit_mlp_accuracy = standard_mlp(model=mlp, test_loader=digit_test_loader, train=False)
        fashion_mlp_accuracy = standard_mlp(model=mlp, test_loader=fashion_test_loader, train=False)

        # # mnist-digit loaders
        digit_train_loader = mnist_dataloader(train_digit_images, train_digit_labels, batch_size=1, shuffle=True)
        digit_test_loader = mnist_dataloader(test_digit_images, test_digit_labels, batch_size=1, shuffle=False)
        # mnist-fashion loaders
        fashion_train_loader = mnist_dataloader(train_fashion_images, train_fashion_labels, batch_size=1, shuffle=True)
        fashion_test_loader = mnist_dataloader(test_fashion_images, test_fashion_labels, batch_size=1, shuffle=False)
        # Neuron train and test
        neuron.runner(train_loader=fashion_train_loader, train=True)
        neuron.runner(train_loader=digit_train_loader, train=True)
        digit_neuron_accuracy = neuron.runner(test_loader=digit_test_loader, train=False)
        fashion_neuron_accuracy = neuron.runner(test_loader=fashion_test_loader, train=False)

        print(f'EPOCH: {epoch}: MLP Digit: {digit_mlp_accuracy} MLP Fashion: {fashion_mlp_accuracy} Neuron Digit: {digit_neuron_accuracy} Neuron Fashion: {fashion_neuron_accuracy}')
        
        # mlp_accuracies.append(mlp_accuracy)
        # agent_accuracies.append(agent_accuracy)

        # print(f'EPOCH: {epoch}: Mlp: {mlp_accuracy} Neuron: {agent_accuracy}')

        # # After training completes:
        # plot_training_progress(mlp_accuracies, agent_accuracies, 'training_progress_v2.png')

training_runner()