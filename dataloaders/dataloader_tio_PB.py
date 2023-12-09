import numpy as np
import torch
import torchio as tio

class MyTransform(tio.transforms.Transform):
    def __init__(self):
        super().__init__()
        self.ct_transform = tio.Compose([
            tio.Clamp(out_min=-1024, out_max=1024),
            tio.RescaleIntensity(out_min_max=(-1,1)),
        ])
        self.pet_transforms = tio.Compose([
            tio.transforms.ZNormalization()        
        ])
        self.label_transforms = None
        
    def stack_images(self, image1, image2, label):
        image = np.stack([image1, image2], axis=-1)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(np.array(label).astype(np.float32))
        return tio.ScalarImage(tensor=image.squeeze(0).permute(3,0,1,2)), tio.LabelMap(tensor=label)
        
    def apply_transform(self, subject):
        subject.ct_image = self.ct_transform(subject.ct_image)
        subject.pet_image = self.pet_transforms(subject.pet_image)
        subject.label = self.label_transforms(subject.label) if self.label_transforms else subject.label
        org_spacing = subject.org_spacing
        id = subject.id
        
        image, label = self.stack_images(subject.ct_image, subject.pet_image, subject.label)
        
        subject.remove_image("ct_image")
        subject.remove_image("pet_image")
        return tio.Subject(image = image, label = label, org_spacing = org_spacing, id = id)

class AutopetDataloaderTioPB():
    def __init__(self, ct_images: str, pet_images: str, labels: str, org_spacing, id) -> None:
        self.ct_images = ct_images
        self.pet_images = pet_images
        # self.suv_images = suv_images
        self.labels = labels
        self.org_spacing = org_spacing
        self.id = id

        assert len(ct_images) == len(pet_images) == len(labels)

    def __getitem__(self, idx):
        subject = tio.Subject(
            ct_image = tio.ScalarImage(self.ct_images[idx]),
            pet_image = tio.ScalarImage(self.pet_images[idx]),
            label = tio.LabelMap(self.labels[idx]),
            org_spacing = self.org_spacing[idx],
            id = self.id[idx]
        )
        
        return subject

    def __len__(self):
        return len(self.ct_images)

#######################################################

class MyTransformBB(tio.transforms.Transform):
	def __init__(self):
		super().__init__()
	
	def stack_images(self, image1, image2, label):
		image = np.stack([image1, image2], axis=-1)
		image = torch.from_numpy(image.astype(np.float32))
		label = torch.from_numpy(np.array(label).astype(np.float32))
		return tio.ScalarImage(tensor=image.squeeze(0).permute(3,0,1,2)), tio.LabelMap(tensor=label)

	def apply_transform(self, subject):
		
		image, label = self.stack_images(subject.ct_image, subject.pet_image, subject.label)

		return tio.Subject(image = image, label = label)

class AutopetDataloaderTioPBBB():
    def __init__(self, ct_images: str, pet_images: str, labels: str) -> None:
        self.ct_images = ct_images
        self.pet_images = pet_images
        self.labels = labels

        assert len(ct_images) == len(pet_images) == len(labels)

    def __getitem__(self, idx):
        subject = tio.Subject(
            ct_image = tio.ScalarImage(self.ct_images[idx]),
            pet_image = tio.ScalarImage(self.pet_images[idx]),
            label = tio.LabelMap(self.labels[idx]),
        )
        return subject

    def __len__(self):
        return len(self.ct_images)

#######################################################

class MyTransform3CH(tio.transforms.Transform):
    def __init__(self):
        super().__init__()

        self.ct_transform = tio.Compose([
            tio.Clamp(out_min=-1024, out_max=1024),
            tio.RescaleIntensity(out_min_max=(-1, 1)),
        ])
        self.pet_transforms = tio.Compose([
            tio.transforms.ZNormalization()        
        ])
        self.output_image_transforms = None
        self.label_transforms = None

    def stack_images(self, image1, image2, image3, label):
        image = np.stack([image1, image2, image3], axis=-1)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(np.array(label).astype(np.float32))
        return tio.ScalarImage(tensor=image.squeeze(0).permute(3,0,1,2)), tio.LabelMap(tensor=label)

    def apply_transform(self, subject):
        subject.ct_image = self.ct_transform(subject.ct_image)
        subject.pet_image = self.pet_transforms(subject.pet_image)
        subject.output_image = self.output_image_transforms(subject.output_image) if self.output_image_transforms else subject.output_image
        subject.label = self.label_transforms(subject.label) if self.label_transforms else subject.label
        org_spacing = subject.org_spacing
        id = subject.id

        image, label = self.stack_images(subject.ct_image, subject.pet_image, subject.output_image, subject.label)

        subject.remove_image("ct_image")
        subject.remove_image("pet_image")
        subject.remove_image("output_image")
        return tio.Subject(image=image, label=label, org_spacing = org_spacing, id = id)


class AutopetDataloaderTioPB3CH():
    def __init__(self, ct_images: str, pet_images: str, labels: str, output_images, org_spacing, id) -> None:
        self.ct_images = ct_images
        self.pet_images = pet_images
        self.output_images = output_images
        self.labels = labels
        self.org_spacing = org_spacing
        self.id = id

        assert len(ct_images) == len(pet_images) == len(labels)

    def __getitem__(self, idx):
        subject = tio.Subject(
            ct_image = tio.ScalarImage(self.ct_images[idx]),
            pet_image = tio.ScalarImage(self.pet_images[idx]),
            output_image = tio.ScalarImage(self.output_images[idx]),
            label = tio.LabelMap(self.labels[idx]),
            org_spacing = self.org_spacing[idx],
            id = self.id[idx]
        )
        return subject

    def __len__(self):
        return len(self.ct_images)

#################################################################################

class MyTransformEvalBB(tio.transforms.Transform):
	def __init__(self):
		super().__init__()
	
	def stack_images(self, image1, image2, label):
		image = np.stack([image1, image2], axis=-1)
		image = torch.from_numpy(image.astype(np.float32))
		label = torch.from_numpy(np.array(label).astype(np.float32))
		return tio.ScalarImage(tensor=image.squeeze(0).permute(3,0,1,2)), tio.LabelMap(tensor=label)

	def apply_transform(self, subject):
		
		image, label = self.stack_images(subject.ct_image, subject.pet_image, subject.label)
		return tio.Subject(image=image, label=label, label_path=subject.label_path, boxes=subject.boxes,  org_spacing = subject.org_spacing, id = subject.id, ct_image_path = subject.ct_image_path )

class AutopetDataloaderTioPBEvalBB():
    def __init__(self, ct_images: str, pet_images: str, labels: str, label_path, boxes, org_spacing, id, ct_image_path) -> None:
        self.ct_images = ct_images
        self.pet_images = pet_images
        self.labels = labels
        self.label_path = label_path
        self.boxes = boxes
        self.org_spacing = org_spacing
        self.id = id
        self.ct_image_path = ct_image_path

        assert len(ct_images) == len(pet_images) == len(labels) == len(label_path) == len(boxes)

    def __getitem__(self, idx):
        subject = tio.Subject(
            ct_image = tio.ScalarImage(self.ct_images[idx]),
            pet_image = tio.ScalarImage(self.pet_images[idx]),
            label = tio.LabelMap(self.labels[idx]),
            label_path = self.label_path[idx],
            boxes = self.boxes,
            org_spacing = self.org_spacing[idx],
            id = self.id[idx],
            ct_image_path = self.ct_images[idx]
        )
        return subject

    def __len__(self):
        return len(self.ct_images)