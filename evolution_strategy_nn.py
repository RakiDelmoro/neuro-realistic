import torch
import pickle
import numpy as np
import torch.nn as nn

class NeuralNetwork:
    def __init__(self, param_shape=(784, 10), latent_dim=5, learning_rate=0.001):
        self.param_shape = param_shape
        self.d = param_shape[0] * param_shape[1]
        self.k = latent_dim
        self.lr = learning_rate
        self.weights = torch.empty(*param_shape, device='cuda').uniform_(-1, 1)
    
    def apply_perturbation(self, num_candidates, step_size=0.01):
        # Shape: (num_candidates, input_size, latent_dim) and (num_candidates, output_size, latent_dim)
        a_matrices = torch.randn(num_candidates, self.param_shape[0], self.k, device='cuda')
        b_matrices = torch.randn(num_candidates, self.param_shape[1], self.k, device='cuda')
        # (num_candidates,input_size,latent_dim), (num_candidates,latent_dim,output_size) -> (num_candidates,input_size,output_size)
        perturbations = torch.bmm(a_matrices, b_matrices.transpose(1, 2))
        # Add to main weights
        candidates = self.weights.unsqueeze(0) + step_size * perturbations

        return candidates, perturbations

    def forward_pass(self, batched_image, batched_labels, candidates):
        # (batch,input_size), (num_candidates,input_size,output_size) -> (num_candidates,batch,output_size)
        candidates_logits = torch.einsum('bi,nio->nbo', batched_image, candidates)
        labels_expanded = batched_labels.unsqueeze(0).expand(candidates.shape[0], -1)
        logits_flattened = candidates_logits.reshape(-1, candidates_logits.shape[-1])
        labels_flattened = labels_expanded.reshape(-1)
        loss_flattened = nn.functional.cross_entropy(logits_flattened, labels_flattened, reduction='none')
        loss_per_candidate = loss_flattened.reshape(candidates.shape[0], -1).mean(dim=-1)
        candidates_reward = -loss_per_candidate

        return candidates_reward

    def update_weights(self, perturbations, rewards):
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        gradient = (rewards_normalized.view(-1, 1, 1) * perturbations).mean(dim=0)
        self.weights += self.lr * gradient

    def predict(self, test_image):
        logits = torch.matmul(test_image, self.weights)
        predictions = torch.argmax(logits, dim=-1)
        return predictions.item()

def mnist_dataloader(img_arr, label_arr, batch_size, shuffle):
    num_samples = img_arr.shape[0]    
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        batched_img = img_arr[indices[start_idx:end_idx]]
        batched_label = label_arr[indices[start_idx:end_idx]]

        yield batched_img, batched_label

def runner():
    EPOCHS = 100
    BATCH_SIZE = 1
    NUM_CANDIDATES = 1000
    with open('./dataset/mnist-digits.pkl', 'rb') as f: ((train_digit_images, train_digit_labels), (test_digit_images, test_digit_labels), _) = pickle.load(f, encoding='latin1')

    model = NeuralNetwork(param_shape=(784,10), latent_dim=5)
    for epoch in range(EPOCHS):
        # Dataloader
        train_loader = mnist_dataloader(train_digit_images, train_digit_labels, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = mnist_dataloader(test_digit_images, test_digit_labels, batch_size=BATCH_SIZE, shuffle=False)

        # Training Loop
        for train_image, train_label in train_loader:
            image = torch.tensor(train_image, device='cuda')
            label = torch.tensor(train_label, device='cuda')
            possible_candidates, perturbations = model.apply_perturbation(num_candidates=NUM_CANDIDATES)
            candidates_reward = model.forward_pass(image, label, possible_candidates)
            model.update_weights(perturbations, candidates_reward)
        # Test Loop
        correctness = []
        for test_image, test_label in test_loader:
            test_image = torch.tensor(test_image, device='cuda')
            test_label = torch.tensor(test_label, device='cuda')
            predicted = model.predict(test_image)
            correctness.append(predicted == test_label.item())
        accuracy = 0.0 if sum(correctness) == 0 else sum(correctness) / len(correctness)

        print(f'EPOCH: {epoch+1} Accuracy: {accuracy}')

runner()
