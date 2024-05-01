import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator
from matplotlib import patches
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_images(images, titles, pixel_scale, gamma=0.45, vmax_factors=None):
    """
    Plots a variable number of images side by side with customizable settings.

    Parameters:
    - images : list of ndarray
      A list of 2D numpy arrays, each containing the image data.
    - titles : list of str
      Titles for each subplot.
    - pixel_scale : float
      The pixel scale in arcseconds per pixel.
    - gamma : float
      Gamma value for PowerNorm.
    - vmax_factors : list of float
      List of factors to calculate the maximum value for normalization for each image. If None, uses a default of 0.2 for each.
    """

    if vmax_factors is None:
        vmax_factors = [0.2] * len(images)  # Default vmax factor

    # Calculate boundaries based on the first image dimensions and pixel scale
    size_y, size_x = images[0].shape
    hmin = (-size_x/2 * pixel_scale, -size_y/2 * pixel_scale)
    hmax = (size_x/2 * pixel_scale, size_y/2 * pixel_scale)

    # Plot setup
    shape = np.array(images[0].shape)
    dpi = 96
    plotsize = shape / dpi
    mmin = np.array((0.45, 0.3))
    mmax = np.array((0.05, 0.45))
    space = np.array((0.45, 0))
    figsize = 0.3 * plotsize * np.array((len(images), 1)) + mmin + mmax + 3 * space
    fig, axes = plt.subplots(1, len(images), figsize=figsize, dpi=dpi)
    plt.subplots_adjust(left=mmin[0]/figsize[0], bottom=mmin[1]/figsize[1], right=1.0-mmax[0]/figsize[0], top=1.0-mmax[1]/figsize[1], wspace=space[0] / figsize[0], hspace=space[1] / figsize[1])

    if len(images) == 1:
        axes = [axes]  # Make single subplot iterable

    # Normalize and plot each image
    for ax, img, title, vmax_factor in zip(axes, images, titles, vmax_factors):
        pnorm = colors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax_factor * np.max(img))
        ax.imshow(img, cmap='gray', norm=pnorm, extent=[hmin[0], hmax[0], hmin[1], hmax[1]])
        ax.set_title(title, color='white')

        # Configure axes aesthetics
        ticklabel = "%g\""
        ax.xaxis.set_major_formatter(FormatStrFormatter(ticklabel))
        ax.yaxis.set_major_formatter(FormatStrFormatter(ticklabel))
        ax.tick_params(axis='both', colors='white', labelsize=14, direction='in')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2)
        ax.set_facecolor('black')
    
    fig.set_facecolor('black')
    plt.tight_layout()
    plt.show()


def plot_image_comparison_zoom(images, titles, pixel_scale, vmax_factors, zoom_rect):
    """
    Plots multiple images with zoomed regions and customization for an arbitrary number of images.

    Parameters:
    - images: list of numpy arrays
      List containing the image data for each subplot.
    - titles: list of str
      Titles for each image.
    - pixel_scale: float
      Pixel scale in arcseconds per pixel.
    - vmax_factors : list of float
      List of factors to calculate the maximum value for normalization for each image. If None, uses a default of 0.2 for each.
    - zoom_rect: list
      Zoomed-in region coordinates and size [x, y, width, height].
    """
    if not images:
        raise ValueError("No images provided")

    if vmax_factors is None:
        vmax_factors = [0.2] * len(images)  # Default vmax factor

    image_height, image_width = images[0].shape

    # Define the coordinates for the full images based on the pixel scale
    image_height, image_width = images[0].shape
    hmin = (-(image_width / 2) * pixel_scale, -(image_height / 2) * pixel_scale)
    hmax = ((image_width / 2) * pixel_scale, (image_height / 2) * pixel_scale)
    ticklabel = "%g\""

    # Create the figure and subplots
    nrows = len(images)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 12))

    for i, (image, vmax_factor, title) in enumerate(zip(images, vmax_factors, titles)):
        # Create normalization instance for each image
        pnorm = colors.PowerNorm(gamma=0.45, vmin=0, vmax=vmax_factor * np.max(image))

        # Plot the full image
        axes[i, 0].imshow(image, cmap='gray', norm=pnorm, extent=[hmin[0], hmax[0], hmin[1], hmax[1]])
        axes[i, 0].set_title(title, color='white', fontweight='bold', size=16)

        # Calculate zoomed region in pixel coordinates
        def coord_to_pixel(coord, min_val, max_val, total_pixels):
            return int((coord - min_val) / (max_val - min_val) * total_pixels)

        y_pixel_start = coord_to_pixel(zoom_rect[1] - zoom_rect[3], hmin[1], hmax[1], image.shape[0])
        y_pixel_end = coord_to_pixel(zoom_rect[1], hmin[1], hmax[1], image.shape[0])
        x_pixel_start = coord_to_pixel(zoom_rect[0], hmin[0], hmax[0], image.shape[1])
        x_pixel_end = coord_to_pixel(zoom_rect[0] + zoom_rect[2], hmin[0], hmax[0], image.shape[1])

        zoom_slice = (slice(y_pixel_start, y_pixel_end), slice(x_pixel_start, x_pixel_end))

        # Plot the zoomed-in region
        axes[i, 1].imshow(image[zoom_slice], cmap='gray', norm=pnorm, extent=[0, zoom_rect[2], 0, zoom_rect[3]])

        # Draw rectangles on the full images to indicate the zoomed region
        rect = patches.Rectangle((zoom_rect[0], zoom_rect[1]), zoom_rect[2], zoom_rect[3], linewidth=4, edgecolor='r', facecolor='none')
        axes[i, 0].add_patch(rect)

        # Connect the original and zoomed images
        connection1 = patches.ConnectionPatch(xyA=(zoom_rect[0] + zoom_rect[2], zoom_rect[1]), xyB=(0, 0), coordsA=axes[i, 0].transData, coordsB=axes[i, 1].transData, color="red", lw=4)
        connection2 = patches.ConnectionPatch(xyA=(zoom_rect[0] + zoom_rect[2], zoom_rect[1] + zoom_rect[3]), xyB=(0, zoom_rect[3]), coordsA=axes[i, 0].transData, coordsB=axes[i, 1].transData, color="red", lw=4)
        fig.add_artist(connection1)
        fig.add_artist(connection2)

        # Customize axes and spines
        for ax in axes.ravel():
            ax.tick_params(axis='both', colors='white', labelsize=14, direction='in')
            ax.xaxis.set_major_formatter(FormatStrFormatter(ticklabel))
            ax.yaxis.set_major_formatter(FormatStrFormatter(ticklabel))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.set_facecolor('black')
            # Default spine color is white
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(2)
        
        # Set red spines only for axes in the second column
        for j in range(nrows):
            for spine in axes[j, 1].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(4)

    fig.set_facecolor('black')
    plt.tight_layout()
    plt.show()


def plot_precision_recall(recall_avg, precision_avg, thresh_avg, recall_infer, precision_infer, thresh_infer, figsize=(5,5)):
    """
    Plots a precision-recall scatter plot for average and inferred frame data.

    Parameters:
    - recall_avg: list of float
      Recall values for average frames.
    - precision_avg: list of float
      Precision values for average frames.
    - thresh_avg: list of float
      Threshold values for average frames.
    - recall_infer: list of float
      Recall values for inferred frames.
    - precision_infer: list of float
      Precision values for inferred frames.
    - thresh_infer: list of float
      Threshold values for inferred frames.
    - figsize: tuple of int
      Figure size.
    """
    # Convert lists to numpy arrays if not already np.ndarray
    if not isinstance(recall_avg, np.ndarray):
        recall_avg = np.array(recall_avg)
    if not isinstance(precision_avg, np.ndarray):
        precision_avg = np.array(precision_avg)
    if not isinstance(thresh_avg, np.ndarray):
        thresh_avg = np.array(thresh_avg)
    if not isinstance(recall_infer, np.ndarray):
        recall_infer = np.array(recall_infer)
    if not isinstance(precision_infer, np.ndarray):
        precision_infer = np.array(precision_infer)
    if not isinstance(thresh_infer, np.ndarray):
        thresh_infer = np.array(thresh_infer)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Scatter plot for average and inferred frames
    sc_avg = ax.scatter(recall_avg*100, precision_avg*100, marker='o', label='30s. avg. frame', c=thresh_avg, cmap='Blues')
    sc_infer = ax.scatter(recall_infer*100, precision_infer*100, marker='D', label='30s. inferred frame', c=thresh_infer, cmap='Oranges')

    # Axis labels
    ax.set_xlabel('\% of stars recovered', size=20)
    ax.set_ylabel('Precision (\%)', size=20)
    
    # Add colorbar
    cbaxes = inset_axes(ax, width="40%", height="3%", bbox_to_anchor=(.5, .07, .6, .5), bbox_transform=ax.transAxes, loc=8)
    cb = fig.colorbar(sc_infer, cax=cbaxes, orientation='horizontal')
    cb.ax.set_title('Detection\nThreshold', size=15)
    cb.ax.tick_params(labelsize=15)
    for label in cb.ax.get_xaxis().get_ticklabels():
        label.set_size(8)

    # Custom legend
    line1 = pylab.Line2D(range(1), range(1), color='white', marker='D', markersize=10, markerfacecolor="orange", alpha=1.0)
    line2 = pylab.Line2D(range(1), range(1), color='white', marker='o', markersize=10, markerfacecolor="blue", alpha=1.0)
    ax.legend((line1, line2), ('30s. inferred frame', '30s. avg. frame'), numpoints=1, loc='lower left', fontsize=12, frameon=False)

    #ax.set_ylim((93, 102))  # Set the y-axis limits

    plt.tight_layout()
    plt.show()