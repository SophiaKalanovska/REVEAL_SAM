import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tensorflow as tf
primes = np.array([ 3,   5,    11,  13,  17,  19,  23,  29,  31,  37, 41, 43])
    # , 
    #     41,  43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,
    #    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
    #    167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    #    239, 241, 251, 257, 263, 269, 271, 277

color_map = {0: np.array([255, 255, 255]), # white
             3: np.array([204, 0, 0]), #dark red 
             5: np.array([255, 0, 0]), # red
            #  7: np.array([255, 102, 102]), # light red
             11: np.array([255,128,0]),# orange
             13: np.array([255,178,102]),# light orange
             17: np.array([255, 255, 0]),# yellow
             19: np.array([0,204,0]), # green
             23: np.array([51, 255, 153]), # blueish green
             29: np.array([0, 128, 255]), # light blue
             31: np.array([0, 0, 204]),  # blue
             37: np.array([0, 0, 153]), # dark blue
             41: np.array([51, 0, 102]), # purple
             43: np.array([0, 0, 0]), #black
               # 41: np.array([255, 180, 80]),
               # 43: np.array([255, 180, 10]),
               # -999: np.array([255, 180, 255]),
              } # light red

def mask_to_input_relevance_of_mask(relevance, masks_from_heatmap3D, label, scene):
    opacity = 0.5
    print(relevance)
    full_relevance = relevance[-1]
    regions_relevance = relevance[-2]
    relevances_clusters = relevance[:-2]
    masks = masks_from_heatmap3D[:-2]
    
    relevances_sorted, masks_with_ones_sorted = zip(*sorted(zip(relevances_clusters, masks), key = lambda x:x[0], reverse=True))

    masked_heat_show = np.zeros_like(masks_with_ones_sorted[0])
    custom_lines = []


    # for i in np.flip(np.argsort(detections.area)):
    #     class_id = (
    #         detections.class_id[i] if detections.class_id is not None else None
    #     )
    #     idx = class_id if class_id is not None else i
    #     color = (
    #         self.color.by_idx(idx)
    #         if isinstance(self.color, ColorPalette)
    #         else self.color
    #     )

    #     mask = detections.mask[i]
    #     colored_mask = np.zeros_like(x, dtype=np.uint8)
    #     colored_mask[:] = color.as_bgr()

    #     scene = np.where(
    #         np.expand_dims(mask, axis=-1),
    #         np.uint8(opacity * colored_mask + (1 - opacity) * scene),
    #         scene,
    #     )
    
    
    for i in range(len(masks_with_ones_sorted)):
        if relevances_sorted[i] > 0:
            mask = masks_with_ones_sorted[i]
            relevance = primes[i]
            masked_heat_show = masked_heat_show * np.logical_not(mask)
            masked_heat_show += relevance * mask
            # custom_lines.append(Line2D([0], [0], color= '#%02x%02x%02x' % tuple(color_map[relevance]), lw=1))
            custom_lines.append(Line2D([0], [0], marker='o', color='w', label='Scatter',
                   markerfacecolor= '#%02x%02x%02x' % tuple(color_map[relevance]), markersize=10))
            color = color_map[relevance]
            colored_mask = np.zeros_like(scene, dtype=np.uint8)
            colored_mask[:] = color.as_bgr()

            scene = np.where(
                np.expand_dims(mask, axis=-1),
                np.uint8(opacity * colored_mask + (1 - opacity) * scene),
                scene,
            )
        else:
            index = i - len(masks_with_ones_sorted)
            mask = masks_with_ones_sorted[index]
            relevance = primes[index]
            masked_heat_show = masked_heat_show * np.logical_not(mask)
            masked_heat_show += relevance * mask
            # custom_lines.append(Line2D([0], [0], color= '#%02x%02x%02x' % tuple(color_map[relevance]), lw=1))
            custom_lines.append(Line2D([0], [0], marker='o', color='w', label='Scatter',
                   markerfacecolor= '#%02x%02x%02x' % tuple(color_map[relevance]), markersize=10))
            
            color = color_map[relevance]
            colored_mask = np.zeros_like(scene, dtype=np.uint8)
            colored_mask[:] = color.as_bgr()

            scene = np.where(
                np.expand_dims(mask, axis=-1),
                np.uint8(opacity * colored_mask + (1 - opacity) * scene),
                scene,
            )

        # plt.imshow(masks[i], color=color_map[relevance * mask],
        #           ax=ax,
        #           label=mask_relevances[i])

    # arr = masked_heat_show[:,:,:, 0][0]
    # relevances_sorted = np.expand_dims(relevances_sorted, -1)

    # data_3d = np.ndarray(shape=(arr.shape[0], arr.shape[1], 3), dtype=int)

    # for i in range(0, arr.shape[0]):
    #     for j in range(0, arr.shape[1]):
    #         data_3d[i][j] = color_map[arr[i][j]]

    # plt.imshow(data_3d)
    # # plt.legend(handles=scatter.legend_elements()[0],
    # #            title="relevances")
    # plt.savefig("hey" + "_masked_heat.png")
    # plt.show()

    relevance_percent_out_of_full = ["%.2f%%" % x  for x in np.squeeze(np.around(relevances_sorted / full_relevance * 100, decimals=2), 1)]
    relevance_percent_out_of_regions = ["%.2f%%" % x  for x in np.squeeze(np.around(relevances_sorted / regions_relevance * 100, decimals=2), 1)]
    relevances = np.squeeze(np.around(relevances_sorted, decimals=3), 1)

    # fig, ax = plt.subplots()
    # plt.tick_params(bottom=False, left=False)
    # ax.axes.yaxis.set_ticklabels([])
    # ax.axes.xaxis.set_ticklabels([])
    # l = ax.legend(custom_lines, relevance_percent_out_of_full, title="Contribution to \n classification\n")
    # plt.setp(l.get_title(), multialignment='center')
    # plt.imshow(data_3d)
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # plt.tick_params(bottom=False, left=False)
    # ax.axes.yaxis.set_ticklabels([])
    # ax.axes.xaxis.set_ticklabels([])
    # l = ax.legend(custom_lines, relevance_percent_out_of_regions, title="Relative \n contribution\n")
    # plt.setp(l.get_title(), multialignment='center')
    # plt.imshow(data_3d)
    # plt.show()

    fig, ax = plt.subplots()
    plt.tick_params(bottom=False, left=False)
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    l = ax.legend(custom_lines, relevances, title="Relevance of label \n " + label + "\n out of " + r"$\bf{" "%.3f" % full_relevance + "}$" "\n",
                  loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, fontsize=11,
                  title_fontsize=11, alignment="center")

    plt.setp(l.get_title(), multialignment='center')
    plt.imshow(scene)
    plt.show()