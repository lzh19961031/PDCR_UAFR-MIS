from collections import namedtuple
import math
import torch
import torch as t
import torch.nn as nn
import torchvision


Size = namedtuple('Size', ('w', 'h'))
Vector = namedtuple('Vector', ('x', 'y'))

class ReceptiveField(namedtuple('ReceptiveField', ('offset', 'stride', 'rfsize', 'outputsize', 'inputsize'))):
  def left(self):
    return t.arange(float(self.outputsize.w)) * self.stride.x + self.offset.x
    
  def top(self):
    return t.arange(float(self.outputsize.h)) * self.stride.y + self.offset.y
  
  def hcenter(self):
    return self.left() + self.rfsize.w / 2
    
  def vcenter(self):
    return self.top() + self.rfsize.h / 2
    
  def right(self):
    return self.left() + self.rfsize.w

  def bottom(self):
    return self.top() + self.rfsize.h
  
  def rects(self):
    return [(x, y, self.rfsize.w, self.rfsize.h) for x in self.left().numpy() for y in self.top().numpy()]
  

  def show(self, image=None, axes=None, show=True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if image is None:
      xs = t.arange(self.inputsize.w).unsqueeze(1)
      ys = t.arange(self.inputsize.h).unsqueeze(0)
      image = (xs.remainder(8) >= 4) ^ (ys.remainder(8) >= 4)
      image = image * 128 + 64

    if axes is None:
      (fig, axes) = plt.subplots(1)

    if isinstance(image, t.Tensor):
      image = image.numpy().transpose(-1, -2)
    axes.imshow(image, cmap='gray', vmin=0, vmax=255)

    rect_density = self.stride.x * self.stride.y / (self.rfsize.w * self.rfsize.h)
    rects = self.rects()

    for (index, (x, y, w, h)) in enumerate(rects): 
      marker, = axes.plot(x + w/2, y + w/2, marker='x')
      if index == 0 or index == len(rects) - 1 or t.rand(1).item() < rect_density:
        axes.add_patch(patches.Rectangle((x, y), w, h, facecolor=marker.get_color(), edgecolor='none', alpha=0.5))
        first = False
    axes.set_xlim(self.left().min().item(), self.right().max().item())
    axes.set_ylim(self.top().min().item(), self.bottom().max().item())
    axes.invert_yaxis()
    if show: plt.show()


(x_dim, y_dim) = (-1, -2)  
def receptivefield(net, input_shape, input_mask_shape, device):
  if len(input_shape) < 4:
    raise ValueError('Input shape must be at least 4-dimensional (N x C x H x W).')
  hooks = []
  def insert_hook(module):
    if isinstance(module, (nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d)):
      hook = _passthrough_grad
      if isinstance(module, nn.MaxPool2d):
        hook = _maxpool_passthrough_grad
      hooks.append(module.register_backward_hook(hook))
  net.apply(insert_hook)
  mode = net.training
  net.eval()
  input = t.ones(input_shape, requires_grad=True, device=device)
  mask = t.ones(input_mask_shape, requires_grad=True, device=device)
  output, _, _, _, _, _ = net(input, mask)
  a = []
  number = 0
  for i in range(len(output)):
    if output[i].dim() < 4:
      raise ValueError('Network is fully connected (output should have at least 4 dimensions: N x C x H x W).')
    outputsize = Size(output[i].shape[x_dim], output[i].shape[y_dim])
    if outputsize.w < 2 and outputsize.h < 2:  
      raise ValueError('Network output is too small along spatial dimensions (fully connected).')
    (x1, x2, y1, y2, pos) = _project_rf(input, output[i], device, return_pos=True)
    rfsize = Size(x2 - x1 + 1, y2 - y1 + 1)
    if rfsize[0] > min(input_shape[2], input_shape[3])-1 or rfsize == min(input_shape[2], input_shape[3])-1:
        continue
    else:
        number = number + 1
    (x1o, _, _, _) = _project_rf(input, output[i], device, offset_x=1)
    (_, _, y1o, _) = _project_rf(input, output[i], device, offset_y=1)
    stride = Vector(x1o - x1, y1o - y1)
    if stride.x == 0 and stride.y == 0:  
      raise ValueError('Input tensor is too small relative to network receptive field.')
    offset = Vector(x1 - pos[x_dim] * stride.x, y1 - pos[y_dim] * stride.y)
    for hook in hooks: hook.remove()
    net.train(mode)
    inputsize = Size(input_shape[x_dim], input_shape[y_dim])
    a.append(ReceptiveField(offset, stride, rfsize, outputsize, inputsize))
  return a, number



def _project_rf(input, output, device, offset_x=0, offset_y=0, return_pos=False):
  pos = [0] * len(output.shape) 
  pos[x_dim] = math.ceil(output.shape[x_dim] / 2) - 1 + offset_x
  pos[y_dim] = math.ceil(output.shape[y_dim] / 2) - 1 + offset_y
  out_grad = t.zeros(output.shape)
  out_grad[tuple(pos)] = 1
  if input.grad is not None:
    input.grad.zero_()
  out_grad = out_grad.to(device)
  output.backward(gradient=out_grad, retain_graph=True)
  in_grad = input.grad[0, 0]
  is_inside_rf = (in_grad != 0.0)
  xs = is_inside_rf.any(dim=y_dim).nonzero()
  ys = is_inside_rf.any(dim=x_dim).nonzero()
  if xs.numel() == 0 or ys.numel() == 0:
    raise ValueError('Could not propagate gradient through network to determine receptive field.')
  bounds = (xs.min().item(), xs.max().item(), ys.min().item(), ys.max().item())
  if return_pos:  
    return (*bounds, pos)
  return bounds

def _passthrough_grad(self, grad_input, grad_output):
  if isinstance(grad_input, tuple) and len(grad_input) > 1:
    return (grad_output[0], *grad_input[1:])
  else:  
    return grad_output

def _maxpool_passthrough_grad(self, grad_input, grad_output):
  assert isinstance(self, nn.MaxPool2d)
  if self.dilation != 1 and self.dilation != (1, 1):
    raise ValueError('Dilation != 1 in max pooling not supported.')
  with t.enable_grad():                               
    input = t.ones(grad_input[0].shape, requires_grad=True).cuda()
    output = nn.functional.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode).cuda()
    return t.autograd.grad(output, input, grad_output[0])

def run_test():
  for kw in [1, 2, 3, 5]: 
    for sx in [1, 2, 3]:  
      for px in [1, 2, 3, 5]:  
        (kh, sy, py) = (kw + 1, sx + 1, px + 1)  
        for width in range(kw + sx * 2, kw + 3 * sx + 1):  
          for height in range(width + 1, width + sy + 1):
            net = nn.Conv2d(3, 2, (kh, kw), (sy, sx), (py, px))
            rf = receptivefield(net, (1, 3, height, width))
            assert rf.rfsize.w == kw and rf.rfsize.h == kh
            assert rf.stride.x == sx and rf.stride.y == sy
            assert rf.offset.x == -px and rf.offset.y == -py
