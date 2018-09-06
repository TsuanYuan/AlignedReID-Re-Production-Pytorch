"""
load appearance model file
Quan Yuan
2018-09-05
"""
import os
import os.path as osp
import datetime
import numpy
import pickle
import torch
from torch.autograd import Variable
from torch.nn.parallel import DataParallel

from aligned_reid.model.Model import MGNModel, SwitchClassHeadModel


class AppearanceModelForward(object):
    def __init__(self, model_path, sys_device_ids=(0,)):
        self.im_mean, self.im_std = [0.486, 0.459, 0.408], [0.229, 0.224, 0.225]

        TVTs, TMOs, relative_device_ids = set_devices_for_ml(sys_device_ids)

        model_name = os.path.basename(model_path)
        parts_model = False
        mgn_model = False
        if model_name.find('parts') >= 0:
            parts_model = True
        if model_name.find('mgn') >= 0:
            mgn_model = True
        if mgn_model:
            model = MGNModel()
        else:
            model = SwitchClassHeadModel(parts_model=parts_model)

        self.model_ws = DataParallel(model, device_ids=relative_device_ids[0])
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model_ws.eval()

    def compute_features_on_batch(self, image_batch):
        patches = []
        for image in image_batch:
            patch = image/255.0
            patch = patch - numpy.array(self.im_mean)
            patch = patch / numpy.array(self.im_std).astype(float)
            patch = patch.transpose(2, 0, 1)
            patches.append(patch)
        patches = numpy.asarray(patches)
        global_feats = self.extract_feature(patches)
        return global_feats

    def extract_feature(self, ims):
        # old_train_eval_model = self.model.training
        ims = Variable(torch.from_numpy(ims).float())
        # global_feat, local_feat, logits = self.model(ims)[0]
        global_feat = self.model_ws(ims)[0].data.cpu().numpy()
        l2_norm = numpy.sqrt((global_feat * global_feat + 1e-10).sum(axis=1))
        global_feat = global_feat / (l2_norm[:, numpy.newaxis])
        return global_feat


def find_index(seq, item):
  for i, x in enumerate(seq):
    if item == x:
      return i
  return -1

def set_devices_for_ml(sys_device_ids):
    """This version is for mutual learning.

    It sets some GPUs to be visible and returns some wrappers to transferring
    Variables/Tensors and Modules/Optimizers.

    Args:
      sys_device_ids: a tuple of tuples; which devices to use for each model,
        len(sys_device_ids) should be equal to number of models. Examples:

        sys_device_ids = ((-1,), (-1,))
          the two models both on CPU
        sys_device_ids = ((-1,), (2,))
          the 1st model on CPU, the 2nd model on GPU 2
        sys_device_ids = ((3,),)
          the only one model on the 4th gpu
        sys_device_ids = ((0, 1), (2, 3))
          the 1st model on GPU 0 and 1, the 2nd model on GPU 2 and 3
        sys_device_ids = ((0,), (0,))
          the two models both on GPU 0
        sys_device_ids = ((0,), (0,), (1,), (1,))
          the 1st and 2nd model on GPU 0, the 3rd and 4th model on GPU 1

    Returns:
      TVTs: a list of `TransferVarTensor` callables, one for one model.
      TMOs: a list of `TransferModulesOptims` callables, one for one model.
      relative_device_ids: a list of lists; `sys_device_ids` transformed to
        relative ids; to be used in `DataParallel`
    """
    import os

    all_ids = []
    for ids in sys_device_ids:
        all_ids += ids
    unique_sys_device_ids = list(set(all_ids))
    unique_sys_device_ids.sort()
    if -1 in unique_sys_device_ids:
        unique_sys_device_ids.remove(-1)

    # Set the CUDA_VISIBLE_DEVICES environment variable

    visible_devices = ''
    for i in unique_sys_device_ids:
        visible_devices += '{}, '.format(i)
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

    # Return wrappers

    relative_device_ids = []
    TVTs, TMOs = [], []
    for ids in sys_device_ids:
        relative_ids = []
        for id in ids:
            if id != -1:
                id = find_index(unique_sys_device_ids, id)
            relative_ids.append(id)
        relative_device_ids.append(relative_ids)

        # Models and user defined Variables/Tensors would be transferred to the
        # first device.
        TVTs.append(TransferVarTensor(relative_ids[0]))
        TMOs.append(TransferModulesOptims(relative_ids[0]))
    return TVTs, TMOs, relative_device_ids


def time_str(fmt=None):
  if fmt is None:
    fmt = '%Y-%m-%d_%H:%M:%S'
  return datetime.datetime.today().strftime(fmt)


def load_pickle(path):
  """Check and load pickle object.
  According to this post: https://stackoverflow.com/a/41733927, cPickle and
  disabling garbage collector helps with loading speed."""
  assert osp.exists(path)
  # gc.disable()
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  # gc.enable()
  return ret


def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """
  # This clause has mistakes:
  # if path is None or '':

  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)

def save_pickle(obj, path):
  """Create dir and save file."""
  may_make_dir(osp.dirname(osp.abspath(path)))
  with open(path, 'wb') as f:
    pickle.dump(obj, f, protocol=2)


def save_mat(ndarray, path):
  """Save a numpy ndarray as .mat file."""
  from scipy import io
  io.savemat(path, dict(ndarray=ndarray))


def to_scalar(vt):
  """Transform a length-1 pytorch Variable or Tensor to scalar.
  Suppose tx is a torch Tensor with shape tx.size() = torch.Size([1]),
  then npx = tx.cpu().numpy() has shape (1,), not 1."""
  if isinstance(vt, Variable):
    return vt.data.cpu().numpy().flatten()[0]
  if torch.is_tensor(vt):
    return vt.cpu().numpy().flatten()[0]
  raise TypeError('Input should be a variable or tensor')


def transfer_optim_state(state, device_id=-1):
  """Transfer an optimizer.state to cpu or specified gpu, which means
  transferring tensors of the optimizer.state to specified device.
  The modification is in place for the state.
  Args:
    state: An torch.optim.Optimizer.state
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for key, val in state.items():
    if isinstance(val, dict):
      transfer_optim_state(val, device_id=device_id)
    elif isinstance(val, Variable):
      raise RuntimeError("Oops, state[{}] is a Variable!".format(key))
    elif isinstance(val, torch.nn.Parameter):
      raise RuntimeError("Oops, state[{}] is a Parameter!".format(key))
    else:
      try:
        if device_id == -1:
          state[key] = val.cpu()
        else:
          state[key] = val.cuda(device=device_id)
      except:
        pass


def may_transfer_optims(optims, device_id=-1):
  """Transfer optimizers to cpu or specified gpu, which means transferring
  tensors of the optimizer to specified device. The modification is in place
  for the optimizers.
  Args:
    optims: A list, which members are either torch.nn.optimizer or None.
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for optim in optims:
    if isinstance(optim, torch.optim.Optimizer):
      transfer_optim_state(optim.state, device_id=device_id)


def may_transfer_modules_optims(modules_and_or_optims, device_id=-1):
  """Transfer optimizers/modules to cpu or specified gpu.
  Args:
    modules_and_or_optims: A list, which members are either torch.nn.optimizer
      or torch.nn.Module or None.
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for item in modules_and_or_optims:
    if isinstance(item, torch.optim.Optimizer):
      transfer_optim_state(item.state, device_id=device_id)
    elif isinstance(item, torch.nn.Module):
      if device_id == -1:
        item.cpu()
      else:
        item.cuda(device=device_id)
    elif item is not None:
      print('[Warning] Invalid type {}'.format(item.__class__.__name__))


class TransferVarTensor(object):
  """Return a copy of the input Variable or Tensor on specified device."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, var_or_tensor):
    return var_or_tensor.cpu() if self.device_id == -1 \
      else var_or_tensor.cuda(self.device_id)


class TransferModulesOptims(object):
  """Transfer optimizers/modules to cpu or specified gpu."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, modules_and_or_optims):
    may_transfer_modules_optims(modules_and_or_optims, self.device_id)


def set_devices(sys_device_ids):
  """
  It sets some GPUs to be visible and returns some wrappers to transferring
  Variables/Tensors and Modules/Optimizers.
  Args:
    sys_device_ids: a tuple; which GPUs to use
      e.g.  sys_device_ids = (), only use cpu
            sys_device_ids = (3,), use the 4th gpu
            sys_device_ids = (0, 1, 2, 3,), use first 4 gpus
            sys_device_ids = (0, 2, 4,), use the 1st, 3rd and 5th gpus
  Returns:
    TVT: a `TransferVarTensor` callable
    TMO: a `TransferModulesOptims` callable
  """
  # Set the CUDA_VISIBLE_DEVICES environment variable
  import os
  visible_devices = ''
  for i in sys_device_ids:
    visible_devices += '{}, '.format(i)
  os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
  # Return wrappers.
  # Models and user defined Variables/Tensors would be transferred to the
  # first device.
  device_id = 0 if len(sys_device_ids) > 0 else -1
  TVT = TransferVarTensor(device_id)
  TMO = TransferModulesOptims(device_id)
  return TVT, TMO
