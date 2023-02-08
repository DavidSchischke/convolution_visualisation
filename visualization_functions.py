import torch
from torchvision.io import read_image
from torchvision.transforms import Normalize, GaussianBlur

import matplotlib.pyplot as plt
import matplotlib.axes as plt_axis

import imageio

from data_and_model import LeNet5


def read_sample_image(i: int, reshape: bool = True) -> torch.Tensor:
    """
    Reads a single image from the example_imgs folder.

    Args:
        i (int): The digit for which the corresponding img should be read.
        reshape (bool, optional): If `True`, reshapes to correct shape for
        forward pass. Defaults to True, should only be changed for plotting.

    Returns:
        torch.Tensor: _description_
    """
    path = f"example_imgs/{i}.png"
    img = read_image(path=path).type(torch.float)

    img = Normalize((0.1307,), (0.3081,)).forward(img)

    if reshape:
        # Correct shape for predicition using lenet
        return img.reshape(-1, 1, 28, 28)

    # Correct shape for plotting
    return img


def get_all_activations(layer_name: str) -> list[torch.Tensor]:
    """
    Return all activations for a specific layer as a list

    Args:
        layer (str): Name of layer. See `data_and_model.LeNet5()` for options.

    Returns:
        list[torch.Tensor]: List of size 10, with each element being the
        activation for the corresponding figure on the input image.
    """

    model = LeNet5()
    model.load_state_dict(torch.load("lenet5_trained.pth"))

    activation = []

    def get_activation(name: str):
        """
        Hacky way to use the forward hook proposed by ptrblck.
        https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/3

        Args:
            name (str): The layer for which to extract the actviation.
        """

        def hook(model, input, output):
            out = output.detach()
            out = out.squeeze()
            activation.append(out)

        return hook

    # I don't know a better way to access a layer by name :(
    for name, layer in model.named_modules():
        if name == layer_name:
            layer.register_forward_hook(get_activation(layer_name))
            break
        if name == "relu5":
            # Hacky way: If we reach last layer and don't register
            # the hook, we have an unknown layer specified
            raise ValueError(f"Unknown layer {layer_name}")

    for i in range(10):
        img = read_sample_image(i)
        model(img)

    return activation


def plot_conv_activation(
    input_digit: int,
    layer_name: str,
    blur: bool = False,
    kernel_size: int = 5,
    sigma: float = 2,
    save_plot: bool = False,
    figsize: tuple[int] = (12, 12)
) -> None:
    """
    Plots the output of all convolutions of a certain layer for a given input digit. Can add
    blurring to the convolutions, which can help to identify structures in the convolutions
    (especially in `conv1`).

    Args:
        input_digit (int): The input digit
        layer_name (str): The layer to use (`conv1` or `conv2`).
        blur (bool, optional): Whether to blur the filter activations. Defaults to False.
        kernel_size (int, optional): The size of the kernel if blur is applied (essentially, this is a fixed convolution
        with a Gaussian kernel). Defaults to 5, reduce for more local blur or increase for wider blur.
        sigma (float, optional): The standard deviation to use for the Gaussian kernel. Defaults to 2, which is very slight imho.
        save_plots (bool, optional): If true, plot is saved instead of shown.
        figsize (tuple[int], optional): Size of plot.
    """
    if blur:
        blurrer = GaussianBlur(kernel_size, sigma)

    activations = get_all_activations(layer_name)
    digit_activations = activations[input_digit]

    if layer_name == "conv1":
        nrow = 3
        ncol = 3
    else:
        nrow = 4
        ncol = 5

    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    [ax.axis("off") for ax in axs.ravel()]

    orig_img = read_sample_image(input_digit, reshape=True).squeeze()
    axs[0, 0].imshow(orig_img, cmap="gray")

    curr_col = 1
    curr_row = 0
    for conv in digit_activations:
        if blur:
            # First dimension (color channel) required for blur
            img = conv.unsqueeze(dim = 0)
            img = blurrer.forward(img)
            # but not for plots
            img.squeeze_()
        else:
            # If no blur, no modifications necessary
            img = conv

        axs[curr_row, curr_col].imshow(img, cmap="gray")

        curr_col += 1
        if curr_col == ncol:
            curr_col = 1
            curr_row += 1

    fig.suptitle(f"Layer: {layer_name}")
    fig.tight_layout()
    if save_plot:
        if blur:
            plt.savefig(f"media/{layer_name}/{input_digit}_blur_{kernel_size}_{sigma}.png", dpi=72)
        else:
            plt.savefig(f"media/{layer_name}/{input_digit}.png", dpi=72)
    else:
        plt.show()
    
    plt.close()
    plt.clf()
    plt.cla()


def show_sample_imgs(figsize: tuple[int, int] = (13, 7), save_plot: bool = False) -> None:
    """
    Reads all sample images and plots them in a 2x5 plot. Very basic.

    Args:
        figsize (tuple[int, int], optional): Size of the plot. Defaults to (9, 5).
        save_plot (bool, optional): Whether plot should be saved. Defaults to False.
    """

    def read_img_to_2d_np(i: int):
        img = read_sample_image(i, False)
        return img.numpy()[0]

    imgs = [read_img_to_2d_np(i) for i in range(10)]

    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=figsize)
    for i in range(10):
        ax: plt_axis = axs[i // 5, i % 5]

        ax.imshow(imgs[i], cmap="gray")
        ax.set_title(i)

    fig.tight_layout()
    if save_plot:
        plt.savefig("media/sample_image_plot.png", dpi=72)
    else:
        plt.show()

def check_predictions() -> None:
    """
    Evaluates the model for all input images by printing the
    true label and the predicted label.
    """
    model = LeNet5()
    model.load_state_dict(torch.load("lenet5_trained.pth"))

    def get_prediction(img: torch.Tensor):
        """Forward pass for input and argmax on output"""
        preds = model(img)
        pred_img = torch.argmax(preds)

        return pred_img

    for i in range(10):
        img = read_sample_image(i)
        pred = get_prediction(img)

        print(f"- Correct value: {i}\t Prediction: {pred}")


def make_plots_to_gif(layer: str, blur: bool, kernel_size: int | None = None, sigma: int | None = None) -> None:
    """
    A simple function that takes the generated .png files and converts them
    into gifs using `imageio`. Implementation heavily inspired by 
    https://towardsdatascience.com/how-to-create-a-gif-from-matplotlib-plots-in-python-6bec6c0c952c
    """
    frames = []
    
    base_path = f"media/{layer}/"
    for i in range(10): 
        
        if blur: 
            fname = f"{i}_blur_{kernel_size}_{sigma}"
        else: 
            fname = str(i)
        path = f"{base_path}/{fname}.png"

        img = imageio.v2.imread(path)
        frames.append(img)
    
    if blur: 
        out_name = f"{layer}_blur"
    else: 
        out_name = layer

    imageio.mimsave(f"media/{out_name}.gif", frames, fps = 1)


if __name__ == "__main__":
    import os

    LAYERS = ["conv1", "conv2"]
    
    if not os.path.exists("media"):
        os.mkdir("media")
        [os.mkdir(f"media/{layer}") for layer in LAYERS if os.path]

    show_sample_imgs(save_plot=True)

    for layer in LAYERS:
        for i in range(10):
            plot_conv_activation(input_digit=i, layer_name=layer, save_plot=True) # Saves images
            

            # Again, with blur
            if layer == "conv1":
                plot_conv_activation(
                    input_digit=i,
                    layer_name=layer,
                    blur=True,
                    kernel_size=5,
                    sigma=3,
                    save_plot=True
                )
            else: 
                plot_conv_activation(
                    input_digit=i,
                    layer_name=layer,
                    blur=True,
                    kernel_size=3, # A bit smaller than conv1, so include less area in blurring
                    sigma=3,
                    save_plot=True
                )
        
        # Create gifs from stored images
        make_plots_to_gif(layer, blur = False)
        
        if layer == "conv1": 
            make_plots_to_gif(layer, blur=True, kernel_size=5, sigma=3)
        else: 
            make_plots_to_gif(layer, blur=True, kernel_size=3, sigma=3)
                