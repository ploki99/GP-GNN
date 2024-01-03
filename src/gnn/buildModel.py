import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from torch_geometric.loader import DataLoader
from torchsummary import summary
from tqdm import tqdm

from .modules import EarlyStopping
from .graphNeuralNetwork import GraphNeuralNetwork


class BuildModel:
    """
    Class that build and train a GNN model.
    """

    def __init__(self, model, train_loader, val_loader, model_path, learning_rate=0.001):
        """
        :param model: GNN model object
        :type model: GraphNeuralNetwork
        :param train_loader: training DataLoader
        :type train_loader: DataLoader
        :param val_loader: validation DataLoader
        :type val_loader: DataLoader
        :param model_path: path and name where to save the best model weights
        :type model_path: str
        :param learning_rate: learning rate to use for training
        """

        self.train_losses = []
        self.val_losses = []
        self.model = model
        self.training_loader = train_loader
        self.validation_loader = val_loader
        self.model_path = model_path
        # Define optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Define scheduler to reduce automatically the learning rate
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=20, verbose=True)
        # Link model to device
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(device)
        # Set double precision
        self.model.to(torch.double)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def loss_fn(x, y, index, batch_size):
        """
        Custom loss function used: MSE along each batch and then mean on the batches.
        :param x: torch tensor of computed values of shape (n_item, 1)
        :type x: torch.Tensor
        :param y: torch tensor of real values of shape (n_item, 1)
        :type y: torch.Tensor
        :param index: torch tensor containing the batch division of shape (n_item, )
        :type index: torch.Tensor
        :param batch_size: number of data in the batch
        :type batch_size: int
        :return: computed loss
        """
        src = torch.flatten(torch.square(x - y))
        inp = torch.zeros(batch_size, dtype=torch.float64)
        out = inp.scatter_reduce(0, index, src, reduce="mean", include_self=False)
        return torch.mean(out)

    def __train_one_epoch(self, epoch):
        """
        Train the model for one epoch.
        :param epoch: current epoch
        :type epoch: int
        :return: average loss of the epoch
        """
        running_loss = 0.0
        pbar = tqdm(self.training_loader, ncols=100, file=sys.stdout, desc=f"EPOCH {epoch:3}")
        for data in pbar:
            # Zero your gradients for every batch
            self.optimizer.zero_grad()
            # Make predictions for this batch
            outputs = self.model(data.x, data.edge_index, data.edge_attr)
            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, data.y, data.batch, data.batch_size)
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Update loss
            pbar.set_postfix({"batch loss": f"{loss.item():.2e}"})
            running_loss += loss.item()

        return running_loss / len(self.training_loader)

    def train(self, epochs, patience=20):
        """
        Train the model.
        :param epochs: number of epochs
        :type epochs: int
        :param patience: patience to use for early stopping
        """
        # Reset losses
        self.train_losses = []
        self.val_losses = []
        # Add early stopping
        early_stopping = EarlyStopping(patience=patience)

        # Start training
        best_val_loss = np.Inf

        for epoch in range(epochs):

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.__train_one_epoch(epoch)

            running_val_loss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, val_data in enumerate(self.validation_loader):
                    val_outputs = self.model(val_data.x, val_data.edge_index, val_data.edge_attr)
                    val_loss = self.loss_fn(val_outputs, val_data.y, val_data.batch, val_data.batch_size)
                    running_val_loss += val_loss

            avg_val_loss = running_val_loss / len(self.validation_loader)

            # Reduce learning rate if needed
            self.scheduler.step(avg_val_loss)

            print(f'Current loss - training: {avg_loss:.2e} validation: {avg_val_loss:.2e}\n')

            # Save train and validation losses
            self.train_losses.append(avg_loss)
            self.val_losses.append(avg_val_loss)

            # Track the best performance, and save the model's state
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.model_path)

            # Check early stopping
            if early_stopping(avg_val_loss):
                break

        # Load best model
        self.load_best()

    def load_best(self):
        """
        Load best model from file.
        """
        self.model.load_state_dict(torch.load(self.model_path))

    def summary(self):
        """
        Print the summary of the model.
        """
        d = next(iter(self.training_loader))
        summary(self.model, [d.x, d.edge_index, d.edge_attr])

    def print_all_parameters(self):
        """
        Print all parameter of the model.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    def plot_training(self, log_scale=True):
        """
        Plot train and validation losses.
        :param log_scale: if True, use log scale to plot the losses
        """
        w, h = plt.figaspect(0.5)
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(1, 1, 1)
        x = [i for i in range(len(self.train_losses))]
        ax.plot(x, self.train_losses, color='blue', label="Train loss")
        ax.plot(x, self.val_losses, color='orange', label="Validation loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        if log_scale:
            plt.yscale("log")
        plt.legend(loc="upper right")
        plt.title("Train and validation losses")
        plt.grid()
        plt.show()

    def test(self, testing_loader):
        """
        Compute relative errors on testing set
        :param testing_loader: testing DataLoader
        :type testing_loader: DataLoader
        """
        print("\nTesting graph neural network:")
        num = 1
        for data in testing_loader:
            # Compute model output
            out = self.model(data.x, data.edge_index, data.edge_attr)
            inp = torch.zeros(data.batch_size, dtype=torch.float64)
            # Compute loss value
            square_diff = torch.flatten(torch.square(out - data.y))
            test_loss = inp.scatter_reduce(0, data.batch, square_diff, reduce="mean", include_self=False)
            # Compute mean relative error
            rel_errs = torch.flatten(torch.abs((out - data.y)/data.y))
            mean_err = inp.scatter_reduce(0, data.batch, rel_errs, reduce="mean", include_self=False)

            for i, el in enumerate(zip(test_loss, mean_err)):
                n_vertexes = sum(data.batch == i)
                e1, e2 = el
                print(f"Graph {num:3.0f} with {n_vertexes} nodes, test loss is {e1.item():.2e}, "
                      f"relative error is {e2.item():.2e}")
                num += 1
