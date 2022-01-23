"""Define your architecture here."""
import torch
from torch import nn
import torch.nn.functional as F

def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = Bonus()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model

class Bonus(nn.Module):
    """Simple Convolutional and Fully Connect network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(7, 7))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(7, 7))
        self.conv4 = nn.Conv2d(24, 32, kernel_size=(7, 7))
        self.conv5 = nn.Conv2d(32, 64, kernel_size=(7, 7))

        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)


    def forward(self, image):
        """Compute a forward pass."""
        first_conv_features = self.pool(F.relu(self.conv1(image)))
        second_conv_features = self.pool(F.relu(self.conv2(
            first_conv_features)))
        third_conv_features = self.pool(F.relu(self.conv3(
            second_conv_features)))
        fourth_conv_features = self.pool(F.relu(self.conv4(
            third_conv_features)))
        fifth = self.pool(F.relu(self.conv5(
            fourth_conv_features)))
        # flatten all dimensions except batch
        flattened_features = torch.flatten(fifth, 1)
        fully_connected_first_out = F.relu(self.fc1(flattened_features))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output