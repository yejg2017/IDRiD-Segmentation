import os
import argparse
import torch
import numpy as np

from custom_scripts.tools.utils import update_lr, get_optimizer
from tensorboardX import SummaryWriter
from custom_scripts.dataset.Diabetic import MulDiabeticDataset,Iterator
from custom_scripts.tools.loss import DICELossMultiClass
from custom_scripts.models.Unet_Series import AttU_Net,U_Net,R2AttU_Net,R2U_Net
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        """DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs""")
    parser.add_argument("--image_size", type=int, default=224, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=4, help="The number of images per batch")
    parser.add_argument("--model",type=str,default="AttU_Net",choices=["AttU_Net","U_Net","R2U_Net","R2AttU_Net"])
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epoches", type=int, default=10)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")

    parser.add_argument("--every_save",type=int,default=5,help="each step to save model")
    parser.add_argument("--dataset", type=str, default="IDRiD_Diabetic", choices=["IDRiD_Diabetic"],
                        help="The dataset used")
    parser.add_argument("--data_path", type=str, default="D:\\DataSet\\Image\\Indian-Diabetic\\Segmentation", help="the root folder of dataset")
    parser.add_argument("--pre_trained_model", type=str, default=None)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # training_params = {"batch_size": opt.batch_size,
    #                    "shuffle": True,
    #                    "drop_last": True,
    #                    "collate_fn": custom_collate_fn}
    #
    # test_params = {"batch_size": opt.batch_size,
    #                "shuffle": False,
    #                "drop_last": False,
    #                "collate_fn": custom_collate_fn}

    training_set =MulDiabeticDataset(root=opt.data_path,is_training=True,image_size=opt.image_size)
    training_set.prepare()
    training_generator =Iterator(training_set,minibatch_size=opt.batch_size)

    test_set = MulDiabeticDataset(root=opt.data_path,is_training=False,image_size=opt.image_size)
    test_set.prepare()
    test_generator = Iterator(test_set,minibatch_size=opt.batch_size)

    #model = Deeplab(num_classes=len(training_set.classes))
    if opt.model=="AttU_Net":
        model=AttU_Net(img_ch=3,output_ch=len(training_set.classes))
    elif opt.model=="U_Net":
        model=U_Net(img_ch=3,output_ch=len(training_set.classes))
    elif opt.model=="R2U_Net":
        model=R2U_Net(img_ch=3,output_ch=len(training_set.classes))
    elif opt.model=="R2AttU_Net":
        model=R2AttU_Net(img_ch=3,output_ch=len(training_set.classes))
    else:
        raise ValueError("There is not this model")

    if opt.pre_trained_model is not None:
        model.load_state_dict(torch.load(opt.pre_trained_model))

    criterion=DICELossMultiClass()

    log_path = os.path.join(opt.log_path, "{}".format(opt.dataset))
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)

    saved_path=os.path.join(opt.saved_path,"{}".format(opt.dataset))
    if os.path.isdir(saved_path):
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)

    writer = SummaryWriter(log_path)
    writer.add_graph(model, torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))
    if torch.cuda.is_available():
        model.cuda()

    best_loss = 1e10
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_set.image_ids)//opt.batch_size
    for epoch in range(opt.num_epoches):
        for step in range(num_iter_per_epoch):
            image,mask=training_generator.next_minibatch()
            image=np.transpose(image.astype(np.float32),(0,3,1,2))
            mask=np.transpose(mask.astype(np.float32),(0,3,1,2))

            current_step = epoch * num_iter_per_epoch + step
            current_lr = update_lr(opt.lr, current_step, num_iter_per_epoch * opt.num_epoches)
            optimizer = get_optimizer(model, current_lr, opt.momentum, opt.decay)
            if torch.cuda.is_available():
                #batch = [torch.Tensor(record).cuda() for record in batch]
                image=torch.from_numpy(image).cuda()
                mask=torch.from_numpy(mask).cuda().float()
            else:
                #batch = [torch.Tensor(record) for record in batch]
                image=torch.from_numpy(image)
                mask=torch.from_numpy(mask).float()

            #image = image.long()
            #mask= mask.float()
            optimizer.zero_grad()
            #results = model(image)
            output=model(image)


            loss=criterion(output,mask)
            loss.backward()
            optimizer.step()

            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {:.2f}".format(
                epoch+1,
                opt.num_epoches,
                step+1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss
            ))

            writer.add_scalar('Train/Total_loss',loss, current_step)

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []


            num_iter_per_epoch_test = len(test_set.image_ids)//opt.batch_size
            for step in range(num_iter_per_epoch_test):
                te_image,te_gt1=test_generator.next_minibatch()
                te_image=np.transpose(te_image.astype(np.float32),(0,3,1,2))
                te_gt1=np.transpose(te_gt1.astype(np.float32),(0,3,1,2))
                if torch.cuda.is_available():
                    te_image=torch.from_numpy(te_image).cuda()
                    te_gt1=torch.from_numpy(te_gt1).cuda().float()
                else:
                    te_image=torch.from_numpy(te_image)
                    te_gt1=torch.from_numpy(te_gt1).float()


                #te_image=te_image.long()
                #te_gt1 = te_gt1.long()
                num_sample = len(te_gt1)

                with torch.no_grad():
                    #te_results = model(te_image)
                    te_output=model(image)
                    #te_mul_losses = multiple_losses(te_results, [te_gt1, te_gt1, te_gt1, te_gt1])
                    te_losses=criterion(te_output,te_gt1)
                #loss_ls.append((te_losses*num_sample))
                loss_ls.append(te_losses)

            te_loss = sum(loss_ls) / test_set.num_images
            print(
                "*** Validation : Epoch: {}/{}, Lr: {}, Loss: {:.2f} ***".format(epoch+1,
                                                            opt.num_epoches,
                                                            optimizer.param_groups[0]['lr'],
                                                            te_loss))
            writer.add_scalar('Test/Total_loss', te_loss, epoch)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model.state_dict(), saved_path + os.sep + "{}_only_params_trained.pth".format(opt.model))
                torch.save(model, saved_path + os.sep + "{}_whole_model_trained_deeplab.pth".format(opt.model))

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break
        if (epoch+1)%opt.every_save==0:
            torch.save(model,os.path.join(saved_path,opt.model,"epoch_"+"{}.pth".format(epoch+1)))
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
