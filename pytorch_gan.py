from torch import nn, optim
import torch
from torch.nn import functional as F
from ai_chaellenger import get_land
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from model.coord_conv import CoordConv, CoordConvTranspose
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super(Down, self).__init__()
        self.cnn = nn.Sequential(
            CoordConv(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU()
        )

    def forward(self, inputs):
        x = self.cnn(inputs)
        return x

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super(Up, self).__init__()
        self.deconv = nn.Sequential(
            CoordConvTranspose(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU()
        )
    def forward(self, x1, x2):
        x = torch.cat([x1, x2],1)
        x = self.deconv(x)
        return x


class Generater(nn.Module):
    def __init__(self):
        super(Generater, self).__init__()
        self.down1 = Down(3, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)
        self.down7 = Down(512, 512)
        self.down8 = Down(512, 512)
        self.decov = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.up1 = Up(1024,512)
        self.up2 = Up(1024, 512)
        self.up3 = Up(1024, 512)
        self.up4 = Up(1024, 256)
        self.up5 = Up(512, 128)
        self.up6 = Up(256, 64)

        self.finnal = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)


    def forward(self, inputs):
        dw1 = self.down1(inputs)
        dw2 = self.down2(dw1)
        dw3 = self.down3(dw2)
        dw4 = self.down4(dw3)
        dw5 = self.down5(dw4)
        dw6 = self.down6(dw5)
        dw7 = self.down7(dw6)
        dw8 = self.down8(dw7)

        decov1 = self.decov(dw8)
        up = self.up1(dw7, decov1)
        up = self.up2(dw6, up)
        up = self.up3(dw5, up)
        up = self.up4(dw4, up)
        up = self.up5(dw3, up)
        up = self.up6(dw2, up)
        up = torch.cat([dw1, up],1)
        logits = self.finnal(up)

        out_put = F.sigmoid(logits)
        return logits, out_put


class Des(nn.Module):
    def __init__(self):
        super(Des, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()

        )

    def forward(self, inputs, target):
        x = torch.cat([inputs, target],1)
        x = self.cnn(x)
        return x


def run():
    EPS = 1e-12
    gen_mod = Generater()

    #gen_mod.load_state_dict(torch.load('log/gen.pth'))
    gan_mod = Des()

    #gan_mod.load_state_dict(torch.load('log/gan.pth'))
    gen_mod.cuda()
    gan_mod.cuda()

    gen_optm = optim.Adam(gen_mod.parameters(), lr=0.002)
    gan_optm = optim.Adam(gan_mod.parameters(), lr=0.002)
    gen_lr= optim.lr_scheduler.StepLR(gen_optm, step_size=30, gamma=0.5)
    gan_lr = optim.lr_scheduler.StepLR(gan_optm, step_size=30, gamma=0.5)

    gen = get_land(batch_size=8, image_size=[256,256])
    for epoch in range(100):
        for step  in range(10000):
            img, org_mask = next(gen)
            count_neg = np.sum(1. - org_mask)
            count_pos = np.sum(org_mask)
            beta = count_neg / (count_neg + count_pos)

            pos_weight = beta / (1 - beta)

            img = np.transpose(img, axes=[0,3,1,2])
            mask = np.transpose(org_mask, axes=[0,3,1,2])
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)


            data, target = torch.autograd.Variable(img.cuda()), torch.autograd.Variable(mask.cuda())
            logits ,outputs = gen_mod(data)
            predict_real = gan_mod(data, target)
            predict_fake = gan_mod(data, outputs)

            discrim_loss = -(torch.log(predict_real + EPS) + torch.log(1 - predict_fake + EPS)).mean()

            gen_loss_GAN = -torch.log(predict_fake + EPS).mean()

            sigmod_loss = F.binary_cross_entropy_with_logits(logits, target)

            gen_loss = gen_loss_GAN * 1 + sigmod_loss * 100

            gen_optm.zero_grad()
            gan_optm.zero_grad()

            discrim_loss.backward(retain_graph=True)

            gen_loss.backward()

            gen_optm.step()
            gan_optm.step()
            d_loss, g_loss, s_loss = discrim_loss.cpu().detach().numpy(), gen_loss_GAN.cpu().detach().numpy(),\
                                     sigmod_loss.cpu().detach().numpy()
            out_put_msk = outputs.cpu().detach().numpy()
            if step%10 ==0:
                print(epoch, d_loss, g_loss, s_loss)

            if step%200==0:
                plt.subplot(121)
                plt.imshow(org_mask[0,:,:,0])
                plt.subplot(122)
                plt.imshow(out_put_msk[0, 0, :,:])
                plt.savefig('dd.jpg')

            if step %1000 ==0:
                torch.save(gen_mod.state_dict(), 'log/'+'gen.pth')
                torch.save(gan_mod.state_dict(), 'log/' + 'gan.pth')
        gen_lr.step(epoch)
        gan_lr.step(epoch)


def eval_test():
    from skimage import io
    EPS = 1e-12
    gen_mod = Generater()
    gen_mod.load_state_dict(torch.load('log/gen.pth'))
    gan_mod = Des()
    gan_mod.load_state_dict(torch.load('log/gan.pth'))
    gen_mod.cuda()
    gan_mod.cuda()
    with torch.no_grad():
        gen_mod.eval()
        gan_mod.eval()

        for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/AIChallenger2018/zuixin/be224/57831739-d102-4f52-aabe-d90d5d0a49d0/*.png'):
            ig = io.imread(x)
            org = ig[:,:,0:3]
            ig = (org-  [123.15, 115.90, 103.06])/255.0
            ig = np.expand_dims(ig, 0)
            ig = np.transpose(ig, [0,3,1,2])
            img = torch.from_numpy(ig).float()
            data = torch.autograd.Variable(img.cuda())
            logits, outputs = gen_mod(data)
            out_put_msk = outputs.cpu().detach().numpy()
            out_put_msk[np.where(out_put_msk>=0.7)] = 1.0
            out_put_msk[np.where(out_put_msk < 0.7)] = 0

            plt.subplot(121)
            plt.imshow(org)
            plt.subplot(122)
            plt.imshow(out_put_msk[0, 0, :, :])
            plt.show()


run()





















