import torch
import torch.nn.functional as F
from .CPD_ResNet_models import CPD_ResNet
import torchvision.transforms as transforms

def get_sod_mask(image, shape = None):
        model = CPD_ResNet()
        model.load_state_dict(torch.load('./CPD/CPD-R.pth'))

        model.cuda()
        model.eval()

        image = image.convert('RGB')

        transform = transforms.Compose([
                transforms.Resize((352, 352)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        image = transform(image).unsqueeze(0)

        image = image.cuda()
        _, res = model(image)
        if shape:
                res = F.upsample(res, size=shape, mode='bilinear', align_corners=False)
        else:
                res = F.upsample(res, size=(240,320), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        return res
