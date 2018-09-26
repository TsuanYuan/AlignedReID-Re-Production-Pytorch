# from pathlib import Path
# import torch
import cv2
# from tqdm import tqdm

def plot_key_points(im_rgb, xs, ys, radius=4, color=(255, 255, 255)):

    count = 0
    for x, y in zip(xs, ys):
        cv2.circle(im_rgb, (int(x), int(y)),
                   radius, color, thickness=2)
        cv2.putText(im_rgb, str(count), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN,
                    1, color, 1)
        count += 1
    return im_rgb

def visualize_keypoints_on_im(im, keypoints_list, title):
    for i, kp_one in enumerate(keypoints_list):
        color = [0]*3
        color[(i+1)%3] = 255
        im = plot_key_points(im, kp_one[:, 0]*im.shape[1], kp_one[:, 1]*im.shape[0], color=tuple(color))
    cv2.imshow(title, im)
    cv2.moveWindow(title, 500 , 200)
    cv2.waitKey()

# class Trainer(object):
#     cuda = torch.cuda.is_available()
#     torch.backends.cudnn.benchmark = True
#
#     def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=5):
#         self.model = model
#         if self.cuda:
#             model.cuda()
#         self.optimizer = optimizer
#         self.loss_f = loss_f
#         self.save_dir = save_dir
#         self.save_freq = save_freq
#
#     def _iteration(self, data_loader, is_train=True):
#         loop_loss = []
#         accuracy = []
#         for data, target in tqdm(data_loader, ncols=80):
#             if self.cuda:
#                 data, target = data.cuda(), target.cuda()
#             output = self.model(data)
#             loss = self.loss_f(output, target)
#             loop_loss.append(loss.data.item() / len(data_loader))
#             accuracy.append((output.data.max(1)[1] == target.data).sum().item())
#             if is_train:
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#         mode = "train" if is_train else "test"
#         print(">>>[{mode}] loss: {sum(loop_loss):.2f}/accuracy: {sum(accuracy) / len(data_loader.dataset):.2%}")
#         return loop_loss, accuracy
#
#     def train(self, data_loader):
#         self.model.train()
#         with torch.enable_grad():
#             loss, correct = self._iteration(data_loader)
#
#     def test(self, data_loader):
#         self.model.eval()
#         with torch.no_grad():
#             loss, correct = self._iteration(data_loader, is_train=False)
#
#     def loop(self, epochs, train_data, test_data, scheduler=None):
#         for ep in range(1, epochs + 1):
#             if scheduler is not None:
#                 scheduler.step()
#             print("epochs: {}".format(ep))
#             self.train(train_data)
#             self.test(test_data)
#             if ep % self.save_freq:
#                 self.save(ep)
#
#     def save(self, epoch, **kwargs):
#         if self.save_dir is not None:
#             model_out_path = Path(self.save_dir)
#             state = {"epoch": epoch, "weight": self.model.state_dict()}
#             if not model_out_path.exists():
#                 model_out_path.mkdir()
#             torch.save(state, model_out_path / "model_epoch_{}.pth".format(epoch))
