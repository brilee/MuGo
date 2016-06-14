import os
import numpy as np

import sgf_wrapper

def load_sgf_positions(*dataset_names):
    for dataset in dataset_names:
        dataset_dir = os.path.join(os.getcwd(), 'data', dataset)
        dataset_files = [os.path.join(dataset_dir, name) for name in os.listdir(dataset_dir)]
        all_datafiles = filter(os.path.isfile, dataset_files)
        for file in all_datafiles:
            with open(file) as f:
                sgf = sgf_wrapper.SgfWrapper(f.read())
                for position_w_context in sgf.get_main_branch():
                    if position_w_context.is_usable():
                        yield position_w_context

def extract_features(features, positions):
    num_feature_planes = sum(f.planes for f in features)
    num_positions = len(positions)
    output = np.zeros([num_positions, 19, 19, num_feature_planes])
    for i, pos in enumerate(positions):
        output[i] = np.concatenate([feature.extract(pos) for feature in features], axis=2)
    return output
