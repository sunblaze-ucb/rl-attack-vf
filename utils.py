import os

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

# Initialize matplotlib.
matplotlib.use('Agg')

COLORMAP = mcolors.ListedColormap(
    (
        (0.10588235294117647, 0.61960784313725492, 0.46666666666666667),
        (0.85098039215686272, 0.37254901960784315, 0.00784313725490196),
        (0.45882352941176469, 0.4392156862745098,  0.70196078431372544),
        (0.4,                 0.65098039215686276, 0.11764705882352941),
        (0.90588235294117647, 0.16078431372549021, 0.54117647058823526),
        (0.90196078431372551, 0.6705882352941176,  0.00784313725490196),
        (0.65098039215686276, 0.46274509803921571, 0.11372549019607843),
        (0.4,                 0.4,                 0.4),
    ),
    'Custom'
)


def plot_decision_boundary(action, index, vis_min, vis_max, vis_steps, b_observations,
                           adversarial_position, b_samples):
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    fig, ax = plt.subplots()
    ax.imshow(b_observations, interpolation='nearest', extent=(vis_min, vis_max, vis_min, vis_max),
              origin='lower', cmap=COLORMAP)

    # Plot position of optimal action and adversarial example.
    ax.plot([0.0], [0.0], marker='x', color='blue')
    ax.plot(adversarial_position, [0.0], marker='s', color='blue')

    # Plot some sample images around the space.
    for xy, image in b_samples.items():
        imagebox = AnnotationBbox(
            OffsetImage(image.reshape([42, 42]), cmap=plt.cm.gray_r, zoom=0.5),
            xy, xycoords='data', annotation_clip=True,
        )
        ax.add_artist(imagebox)

    pid = os.getpid()
    np.save('decision-boundary-{}-{}-{}.npy'.format(pid, index, action), b_observations)
    plt.savefig('decision-boundary-{}-{}-{}.png'.format(pid, index, action), bbox_inches='tight', pad_inches=0)
    plt.savefig('decision-boundary-{}-{}-{}.pdf'.format(pid, index, action), bbox_inches='tight', pad_inches=0)
    plt.close()
