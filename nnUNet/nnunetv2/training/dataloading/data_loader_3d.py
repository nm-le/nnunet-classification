import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hard-coded for UHN dataset
        self.class_samples = {
            0: 62,  
            1: 106,
            2: 84
        }

        total_samples = sum(self.class_samples.values())
        self.class_weights = {
            cls: total_samples / (len(self.class_samples) * count)
            for cls, count in self.class_samples.items()
        }
        
        # Group indices by class
        self.class_indices = {cls: [] for cls in self.class_samples.keys()}
        for idx in self.indices:
            _, _, _, cls = self._data.load_case(idx)
            self.class_indices[cls].append(idx)

    def get_indices(self):
        
        selected_keys = []
        classes_per_batch = self.batch_size // len(self.class_samples)
        remainder = self.batch_size % len(self.class_samples)
        
        for cls in self.class_samples.keys():
            # Sample with replacement if we have fewer samples than needed
            n_samples = classes_per_batch + (1 if remainder > 0 else 0)
            remainder -= 1
            
            if len(self.class_indices[cls]) == 0:
                continue
                
            # Sample indices for this class
            cls_indices = np.random.choice(
                self.class_indices[cls],
                size=n_samples,
                replace=len(self.class_indices[cls]) < n_samples
            )
            selected_keys.extend(cls_indices)
            
        # Shuffle the selected keys to avoid having all samples from the same class
        # appearing consecutively in the batch
        np.random.shuffle(selected_keys)
        
        # If we couldn't fill the batch, fill the remaining slots randomly
        while len(selected_keys) < self.batch_size:
            # Sample from any class
            random_cls = np.random.choice(list(self.class_samples.keys()))
            if len(self.class_indices[random_cls]) > 0:
                random_idx = np.random.choice(self.class_indices[random_cls])
                selected_keys.append(random_idx)
                
        return selected_keys

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        class_labels = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties, cls = self._data.load_case(i)
            case_properties.append(properties)
            class_labels.append(cls)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

                    class_labels = torch.tensor(class_labels, dtype=torch.long)
            return {'data': data_all, 'target': seg_all, 'keys': selected_keys, 'class_labels': class_labels}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys, 'class_labels': class_labels}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
