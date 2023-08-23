import numpy as np

def random_clusters(x, masks = []):
    size_2 = x.shape[1], x.shape[2], 3
    proba_0 = 0.99982
    random_mask_2 = np.expand_dims(np.random.choice([0.0, 1.0], size=size_2, p=[proba_0, 1 - proba_0]), 0)

    zero_mask = np.zeros_like(masks[0])
    one_mask = np.ones_like(masks[0])
    all_regions = np.sum(masks, 0, keepdims=True)[0]
    # masks.append(zero_mask)
    # masks.append(random_mask_2)
    masks.append(all_regions)
    masks.append(one_mask)
    # return [all_regions, one_mask]
    return masks