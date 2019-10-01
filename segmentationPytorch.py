from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import torchvision.models as models
import scipy.misc

from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchsummary import summary
from matplotlib import pyplot as plt

squeezenet = models.squeezenet1_1(pretrained=True)

h, w, c = 500, 1280, 3
nh, nw = int(h/5), int(w/5)

class JuanSqueezeNet(nn.Module):
  def __init__(self):
    # super().__init__()
    super(JuanSqueezeNet, self).__init__()
    self.squeezenet_model = squeezenet.features
    self.layer1 = nn.Sequential(
        nn.ReLU(),
        nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        nn.BatchNorm2d(512),
        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 3, kernel_size=(4,10), stride=1, padding=1, dilation=2),
        nn.BatchNorm2d(3),
    )
    self.classifier = nn.Sigmoid()

  def forward(self, x):
    output = self.squeezenet_model(x)
    output = self.layer1(output)
    output = self.classifier(output.view(len(x),-1))
    return output.view(len(x), c, nh, nw)

net = JuanSqueezeNet().cuda()
net.load_state_dict(torch.load("./model_saved_b-10_e-150_lr-1e-3_s-50_g-0-5.pth"))
net.eval()

def predict(frame, sMask = False):
  frame_torch = cv2.resize(frame, (nw, nh))
  frame_torch = np.moveaxis(frame_torch, 2, 0)
  frame_torch = frame_torch / 255
  # frame_torch = torch.Tensor(frame_torch)

  # plt.imshow(np.moveaxis(frame_torch,0,2))
  # plt.show()

  # frame_torch  = frame_torch.detach().numpy()
  # print(frame_torch)
  mask = net.forward(torch.Tensor([frame_torch]).cuda())[0].cpu().detach().numpy()

  mask = np.moveaxis(mask, 0, 2)
  mask = cv2.resize(mask, (w, h))

  mn = mask.mean()

  if sMask:
    mask[mask < mn] = 0
    mask[mask >= mn] = 1
    cv2.imshow("Mascara",mask)
    while True:
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    # return mask

  segmentation = (mask >= mn)
  segmentation = np.delete(segmentation,2,axis=2)
  segmentation = np.delete(segmentation,1,axis=2)
  
  segmentation = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
  segmentation = scipy.misc.toimage(segmentation, mode="RGBA")
  street_im = scipy.misc.toimage(frame)
  street_im.paste(segmentation, box=None, mask=segmentation)  

  return np.array(street_im)

def predict_video(sMask = False):
  cap = cv2.VideoCapture("./left_output.mp4")

  if not cap.isOpened():
          raise ValueError("ERROR TO LOAD THE VIDEO")

  while cap.isOpened():
    ret,frame = cap.read()
    
    # cv2.imshow("Original", frame)
    cv2.imshow("Imagem Predita", predict(frame[280:], sMask))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()

def predict_img(sMask = True):
  img = cv2.imread("./teste.png")
  cv2.imshow("Imagem Original",img)
  while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cv2.imwrite("./img_predicted.png",predict(img[:], sMask))




########################################################################################################################
# predict_img()
predict_video(False)