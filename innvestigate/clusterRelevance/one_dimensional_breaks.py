from jenkspy import JenksNaturalBreaks
import numpy as np
import jenkspy
import seaborn
import matplotlib.pyplot as plt


def jenks_breaks(activations, numberOfBreaks, image_size, n, cutoff):
    result = activations.flatten()
    image = x
    quaters = np.floor(result.size / n)
    top_nity = np.percentile(result, 90)
    innerbreaks = []
    jnb = JenksNaturalBreaks(numberOfBreaks)
    for i in range(n):
        if int(quaters)*i == 0:
            result_new = result[:int(quaters)]
            jnb.fit(result_new)
            innerbreaks.append(jnb.breaks_)
        elif i == n-1:
            result_new = result[int(quaters) * i:]
            jnb.fit(result_new)
            innerbreaks.append(jnb.breaks_)
        else:
            result_new = result[int(quaters)*i:][:int(quaters)]
            jnb.fit(result_new)
            innerbreaks.append(jnb.breaks_)

    activation_ranges = []
    # activation_new = []
    rows_down = (quaters / activations[1].size) + 1

    y = -1
    for i in activations:
        y += 1
        x = -1
        for j in i:
            x += 1
            for r in range(n):
                if y >= rows_down*(r) and y <= rows_down*(r+1):
                    if abs(j) > innerbreaks[r][-2] and abs(j) > top_nity:
                    # if abs(j) > innerbreaks[r][-3] and abs(j) > 0.012:
                        activation_ranges.append((x , image_size - y, image[0][y][x][0] / 3, image[0][y][x][1] / 3, image[0][y][x][2]/ 3))
                        continue
                        # activation.append(j)
                    #     continue
                    # else:
                    #     activation.append(0)
                    #     continue
        # activation_new.append(activation)


    # activation_ranges = remove_bottom(activation_new, 5, image_size)
    if len(activation_ranges) < cutoff and numberOfBreaks > 1:
        activation_ranges = jenks_breaks(activations, numberOfBreaks - 1, image_size, n, cutoff)
    return activation_ranges


#
# def remove_bottom(activations, numberOfBreaks, image_size):
#     result = np.array(activations).flatten()
#     jnb = JenksNaturalBreaks(numberOfBreaks)
#     jnb.fit(result)
#     innerbreaks = jnb.breaks_
#
#     activation_ranges = []
#
#
#     y = -1
#     for i in activations:
#         y += 1
#         x = -1
#         for j in i:
#             x += 1
#             if j > innerbreaks[1]:
#                 activation_ranges.append((x, image_size - y))
#                 continue
#     #
#     if len(activation_ranges) > 2500:
#         activation_ranges = remove_bottom(activations, numberOfBreaks - 2, image_size)
#     return activation_ranges

def print_jenks_data(jnb):
    try:
        print(jnb.labels_)
        print(jnb.groups_)
        print(np.len(jnb.groups_[0]))
        print(jnb.inner_breaks_)
        print(jnb.breaks_)
    except:
        pass


def visualise_breaks(activations):

    result = activations.flatten()
    breaks = jenkspy.jenks_breaks(result, nb_class=5)
    print(breaks)

    plt.figure()
    seaborn.stripplot(x=result, jitter=True)
    seaborn.despine()
    locs, labels = plt.xticks()
    plt.xticks(locs, map(lambda x: x, np.round(locs,2)))
    plt.xlabel('Intensity')
    plt.yticks([])
    for b in breaks:
        plt.vlines(b, ymin=-0.2, ymax=0.5)

    plt.show()
