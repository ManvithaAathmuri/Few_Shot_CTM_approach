import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from miniImageNet import miniImageNet
from tieredImageNet import tieredImageNet

# Define CTM module with multiple feature extractors
class CTM(nn.Module):
    def __init__(self, in_channels, out_channels, num_support_sets):
        super(CTM, self).__init__()
        self.feature_extractors = nn.ModuleList()
        for _ in range(num_support_sets):
            self.feature_extractors.append(Concentrator(in_channels, out_channels))
        self.projector = Projector(out_channels, out_channels)

    def forward(self, support_images):
        support_features = []
        for feature_extractor in self.feature_extractors:
            support_features.append(feature_extractor(support_images))

        support_features = torch.cat(support_features, dim=0)
        return self.projector(support_features)

# Define MetricLearner module
class MetricLearner(nn.Module):
    def __init__(self, in_channels):
        super(MetricLearner, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# MAML update function
def maml_update(model, loss, alpha):
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Train CTM model with MAML
def train_ctm(support_images, support_labels, query_images, query_labels, model, alpha):
    support_features = model(support_images)
    query_features = model(query_images)

    support_features = support_features.view(support_features.size(0), -1)
    query_features = query_features.view(query_features.size(0), -1)

    similarities = MetricLearner(model.projector.out_channels)(support_features)

    loss = nn.functional.cross_entropy(similarities, query_labels)
    maml_update(model, loss, alpha)

# Evaluate model accuracy
def evaluate(model, support_dataloader, query_dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for support_images, support_labels, query_images, query_labels in support_dataloader:
            query_features = model(query_images)

            query_features = query_features.view(query_features.size(0), -1)

            similarities = MetricLearner(model.projector.out_channels)(query_features)

            _, predicted = torch.max(similarities.data, 1)
            total += query_labels.size(0)
            correct += (predicted == query_labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def forward(self, support_images, query_images):
    support_features = []
    for feature_extractor in self.feature_extractors:
        support_features.append(feature_extractor(support_images))

    support_features = torch.cat(support_features, dim=0)
    query_features = self.feature_extractors[0](query_images)

    support_features = support_features.view(support_features.size(0), -1)
    query_features = query_features.view(query_features.size(0), -1)

    similarities = self.metric_learner(support_features)

    return similarities


# Main function
def main():
    # Load miniImageNet and tieredImageNet datasets
    miniImageNet_support_dataloader, miniImageNet_query_dataloader = miniImageNet(n_way=20, k_shot=5, q_shot=5)
    tieredImageNet_support_dataloader, tieredImageNet_query_dataloader = tieredImageNet(n_way=20, k_shot=5, q_shot=5)

    # Train CTM model using MAML
    model = CTM(in_channels=3, out_channels=64, num_support_sets=2)
    alpha = 0.01

    for epoch in range(10):
        for support_images, support_labels, query_images, query_labels in miniImageNet_support_dataloader:
            train_ctm(support_images, support_labels, query_images, query_labels, model, alpha)

    # Evaluate model on miniImageNet
    miniImageNet_accuracy = evaluate(model, miniImageNet_support_dataloader, miniImageNet_query_dataloader)
    print("miniImageNet Accuracy:", miniImageNet_accuracy)

    # Evaluate model on tieredImageNet
    tieredImageNet_accuracy = evaluate(model, tieredImageNet_support_dataloader, tieredImageNet_query_dataloader)
    print("tieredImageNet Accuracy:", tieredImageNet_accuracy)

if __name__ == "__main__":
    main()
