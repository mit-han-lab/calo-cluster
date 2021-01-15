import numpy as np
import torch
from torchsparse import SparseTensor


def sparse_collate(coords,
                   feats,
                   labels=None,
                   is_double=False,
                   coord_float=False):
    r"""Create a sparse tensor with batch indices C in `the documentation
    <https://stanfordvl.github.io/MinkowskiEngine/sparse_tensor.html>`_.

    Convert a set of coordinates and features into the batch coordinates and
    batch features.

    Args:
        coords (set of `torch.Tensor` or `numpy.ndarray`): a set of coordinates.

        feats (set of `torch.Tensor` or `numpy.ndarray`): a set of features.

        labels (set of `torch.Tensor` or `numpy.ndarray`): a set of labels
        associated to the inputs.

        is_double (`bool`): return double precision features if True. False by
        default.

    """
    use_label = False if labels is None else True
    coords_batch, feats_batch, labels_batch = [], [], []

    batch_id = 0
    for coord, feat in zip(coords, feats):
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord)
        else:
            assert isinstance(
                coord, torch.Tensor
            ), "Coords must be of type numpy.ndarray or torch.Tensor"

        if not coord_float:
            coord = coord.int()
        else:
            coord = coord.float()

        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        else:
            assert isinstance(
                feat, torch.Tensor
            ), "Features must be of type numpy.ndarray or torch.Tensor"
        feat = feat.double() if is_double else feat.float()

        # Batched coords
        num_points = coord.shape[0]

        if not coord_float:
            coords_batch.append(
                torch.cat((coord, torch.ones(num_points, 1).int() * batch_id),
                          1))
        else:
            coords_batch.append(
                torch.cat(
                    (coord, torch.ones(num_points, 1).float() * batch_id), 1))

        # Features
        feats_batch.append(feat)

        # Labels
        if use_label:
            label = labels[batch_id]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            else:
                assert isinstance(
                    label, torch.Tensor
                ), "labels must be of type numpy.ndarray or torch.Tensor"
            labels_batch.append(label)

        batch_id += 1

    # Concatenate all lists
    if not coord_float:
        coords_batch = torch.cat(coords_batch, 0).int()
    else:
        coords_batch = torch.cat(coords_batch, 0).float()
    feats_batch = torch.cat(feats_batch, 0)
    if use_label:
        labels_batch = torch.cat(labels_batch, 0)
        return coords_batch, feats_batch, labels_batch
    else:
        return coords_batch, feats_batch

def sparse_collate_tensors(sparse_tensors):
    coords, feats = sparse_collate([x.C for x in sparse_tensors],
                                   [x.F for x in sparse_tensors])
    return SparseTensor(feats, coords, sparse_tensors[0].s)


def sparse_collate_fn(batch):
    if isinstance(batch[0], dict):
        batch_size = batch.__len__()
        ans_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], SparseTensor):
                ans_dict[key] = sparse_collate_tensors(
                    [sample[key] for sample in batch])
            elif isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = torch.stack(
                    [torch.from_numpy(sample[key]) for sample in batch],
                    axis=0)
            elif isinstance(batch[0][key], torch.Tensor):
                ans_dict[key] = torch.stack([sample[key] for sample in batch],
                                            axis=0)
            elif isinstance(batch[0][key], dict):
                ans_dict[key] = sparse_collate_fn(
                    [sample[key] for sample in batch])
            else:
                ans_dict[key] = [sample[key] for sample in batch]
        ans_dict['subbatch_indices'] = torch.cat([torch.empty(len(list(b.values())[0].C)).fill_(i) for i,b in enumerate(batch)])
        return ans_dict
    else:
        batch_size = batch.__len__()
        ans_dict = tuple()
        for i in range(len(batch[0])):
            key = batch[0][i]
            if isinstance(key, SparseTensor):
                ans_dict += sparse_collate_tensors(
                    [sample[i] for sample in batch]),
            elif isinstance(key, np.ndarray):
                ans_dict += torch.stack(
                    [torch.from_numpy(sample[i]) for sample in batch], axis=0),
            elif isinstance(key, torch.Tensor):
                ans_dict += torch.stack([sample[i] for sample in batch],
                                        axis=0),
            elif isinstance(key, dict):
                ans_dict += sparse_collate_fn([sample[i] for sample in batch]),
            else:
                ans_dict += [sample[i] for sample in batch],
        return ans_dict
