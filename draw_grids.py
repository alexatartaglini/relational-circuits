from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


def imshow(path):
    image = Image.open(path)
    # Set up figure
    my_dpi = 100
    fig = plt.figure(
        figsize=(float(image.size[0]) / my_dpi, float(image.size[1]) / my_dpi),
        dpi=my_dpi,
    )
    ax = fig.add_subplot(111)

    # Remove whitespace from around the image
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    myInterval = 16
    loc = plticker.MultipleLocator(base=myInterval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    # Add the grid
    ax.grid(which="major", axis="both", linestyle="-", color="red")

    # Add the image
    ax.imshow(image)
    plt.savefig("./grids.png", format="png")


path = "stimuli/das/discrimination/trainsize_6400_256-256-256/shape_32/train/same_set_2957/counterfactual.png"
imshow(path)
