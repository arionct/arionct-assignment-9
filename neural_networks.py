import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim=2, hidden_dim=3, output_dim=1, lr=0.1, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        self.hidden_output = None
        self.gradients = None

    def activation(self, x):
        if self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_fn == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("choose other activation function (relu, sigmoid, tanh)")
        
    def activation_derivative(self, x):
        if self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            raise ValueError("choose other activation function (relu, sigmoid, tanh)")
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        z_hidden = np.dot(X, self.W1) + self.b1
        self.hidden_output = self.activation(z_hidden)

        z_output = np.dot(self.hidden_output, self.W2) + self.b2
        out = np.tanh(z_output)
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        output_error = self.forward(X) - y
        d_output = output_error * (1 - self.forward(X) ** 2)

        grad_W2 = np.dot(self.hidden_output.T, d_output)
        grad_b2 = np.sum(d_output, axis=0, keepdims=True)

        hidden_error = np.dot(d_output, self.W2.T)
        if self.activation_fn == 'tanh':
            d_hidden = hidden_error * (1 - self.hidden_output ** 2)
        elif self.activation_fn == 'relu':
            d_hidden = hidden_error * (self.hidden_output > 0)
        elif self.activation_fn == 'sigmoid':
            d_hidden = hidden_error * (self.hidden_output * (1 - self.hidden_output))

        grad_W1 = np.dot(X.T, d_hidden)
        grad_b1 = np.sum(d_hidden, axis=0, keepdims=True)        

        # TODO: update weights with gradient descent
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2

        # TODO: store gradients for visualization
        self.gradients = {
            'input_hidden': grad_W1,
            'hidden_output': grad_W2
        }

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    hidden_layer_output = mlp.hidden_output

    if hidden_layer_output.shape[1] >= 3:
        ax_hidden.scatter(
            hidden_layer_output[:, 0], hidden_layer_output[:, 1], hidden_layer_output[:, 2],
            c=y.ravel(), cmap='bwr', alpha=0.7
        )
        ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
        ax_hidden.set_xlabel("Hidden Dimension 1")
        ax_hidden.set_ylabel("Hidden Dimension 2")
        ax_hidden.set_zlabel("Hidden Dimension 3")

        x_vals = np.linspace(-1.5, 1.5, 50)
        y_vals = np.linspace(-1.5, 1.5, 50)
        xx, yy = np.meshgrid(x_vals, y_vals)
        z_vals = -(mlp.W2[0, 0] * xx + mlp.W2[1, 0] * yy + mlp.b2[0, 0]) / (mlp.W2[2, 0] + 1e-5)
        ax_hidden.plot_surface(xx, yy, z_vals, alpha=0.3, color='tan')
    else:
        raise ValueError("Hidden layer must have at least 3 dimensions for 3D visualization")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)

    ax_input.contour(xx, yy, preds, levels=[0], colors='black', linewidths=1.5)
    ax_input.contourf(xx, yy, preds, levels=[-1, 0, 1], colors=['blue', 'red'], alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k', s=20)
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    ax_input.set_xlabel("Feature 1")
    ax_input.set_ylabel("Feature 2")

    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    ax_gradient.set_xlim(0, 1)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.axis('off')

    nodes = {
        'x1': (0.2, 0.8), 'x2': (0.2, 0.6),
        'h1': (0.5, 0.9), 'h2': (0.5, 0.7), 'h3': (0.5, 0.5),
        'y': (0.8, 0.7)
    }

    for name, (x_pos, y_pos) in nodes.items():
        ax_gradient.add_patch(Circle((x_pos, y_pos), 0.03, color='blue'))
        ax_gradient.text(x_pos, y_pos, name, color='white', ha='center', va='center')

    edges = [ ('x1', 'h1', mlp.gradients['input_hidden'][0, 0]), ('x1', 'h2', mlp.gradients['input_hidden'][0, 1]), ('x1', 'h3', mlp.gradients['input_hidden'][0, 2]), ('x2', 'h1', mlp.gradients['input_hidden'][1, 0]), ('x2', 'h2', mlp.gradients['input_hidden'][1, 1]), ('x2', 'h3', mlp.gradients['input_hidden'][1, 2]), ('h1', 'y', mlp.gradients['hidden_output'][0, 0]), ('h2', 'y', mlp.gradients['hidden_output'][1, 0]), ('h3', 'y', mlp.gradients['hidden_output'][2, 0]) ]

    for start_node, end_node, grad in edges:
        x1, y1 = nodes[start_node]
        x2, y2 = nodes[end_node]
        linewidth = min(3, max(0.5, abs(grad) * 5))
        ax_gradient.plot([x1, x2], [y1, y2], 'm-', linewidth=linewidth)


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 450
    visualize(activation, lr, step_num)