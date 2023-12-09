import torchio as tio

train_transform1 = tio.Compose([
    tio.RandomFlip(0,p=0.4),
    tio.RandomFlip(1,p=0.4),
    tio.RandomFlip(2,p=0.4),
    tio.RandomAffine(scales=(0.8,0.95), degrees=(30), translation=(0,15),p=0.3), 
])

train_transform2 = tio.Compose([
    tio.RandomAffine(scales=(0.7, 1.4), degrees=(-30, 30), translation=0, p=0.4), 
])

train_transform3 = tio.Compose([
    tio.RandomAffine(scales=(0.8, 1.2), degrees=0, translation=0, p=0.4),
    tio.RandomElasticDeformation(p=0.3), 
])

train_transform4 = tio.Compose([
    tio.RandomAffine(scales=(0.8,1.2), degrees=(-30, 30), translation=0, p=0.35),
    tio.RandomNoise(p=0.25),
    tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.25),
])

train_transform5 = tio.Compose([
    tio.RandomFlip(0,p=0.45),
    tio.RandomFlip(1,p=0.45),
    tio.RandomFlip(2,p=0.45),
    tio.RandomAffine(scales=(0.8,0.95), degrees=(35), translation=(0,15),p=0.35), 
    tio.RandomElasticDeformation(p=0.3), 
    tio.RandomNoise(p=0.1),
    tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.1),
])