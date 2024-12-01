import math
import torch
from torch import nn
import pywt
import torch.nn.functional as F
from torchsummary import summary
from einops import rearrange
class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1,groups = 1):
        super(Conv1, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding,groups=groups)
        self.conv = nn.utils.weight_norm(self.conv)
        self.norm = nn.BatchNorm1d(out_channels)
        nn.init.kaiming_normal_(self.conv.weight)
        self.act=nn.ELU()
    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        return out
class WindowPool(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(WindowPool, self).__init__()
        self.pool = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=kernel_size, stride=stride, groups=1)
    def forward(self, x):
        B, C, T = x.size()
        # 通过滑动窗口池化进行时间维度的压缩
        x = rearrange(x, 'b c t ->( b c) 1 t')  # 转换为适应 Conv1d 输入格式
        out = self.pool(x)
 #       B, C, T = out.size()
        x = rearrange(out, "(b c) h t ->b (h c) t",b=B)  # 转换为适应 Conv1d 输入格式
        x = rearrange(x, "b  (h c) t ->b  c (h t)",c = C)  # 转换为适应 Conv1d 输入格式
        return x
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, groups=1, bias=True):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, out_channels, kernel_size, dilation=dilation, padding=self.padding, groups=groups,
                              bias=bias)
        self.window = WindowPool()
        # 权重初始化
        nn.init.kaiming_normal_(self.conv.weight)
        # 如果使用偏置，则初始化偏置为常数0
        if bias:
            nn.init.constant_(self.conv.bias, 0)
        # 使用 Weight Normalization
        self.conv = nn.utils.weight_norm(self.conv)
        # 批归一化和激活函数
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()
    def forward(self, x):
        B, C, T = x.size()
        x0 = self.window(x)
        x =x0+x
        x = rearrange(x, "b (h c) t ->(b c) h t", h=1)
        # 卷积 -> 批归一化 -> 激活
        out = self.act(self.norm(self.conv(x)))
        # 调整输出形状
        out = rearrange(out, "(b c) h t ->b (h c) t", b=B)
        return out

class MultiScalePooling(nn.Module):
    def __init__(self,in_channels, pool_kernels=[75, 115, 155]):
        super(MultiScalePooling, self).__init__()
        # 为每个尺度的池化创建一个池化层
        self.pool_layers = nn.ModuleList([
            Pooling1D(pool_size=k)
            for k in pool_kernels
        ])
        self.conv1x1 = nn.Conv1d(in_channels * len(pool_kernels), in_channels, kernel_size=1, bias=False,groups=1)
    def forward(self, x):
        pooled_outputs = [pool(x) for pool in self.pool_layers]
        # 拼接池化后的结果
  #      concatenated = torch.cat(pooled_outputs, dim=1)
  #      output = concatenated
        return pooled_outputs[0],pooled_outputs[1],pooled_outputs[2]
class GeMP1D(nn.Module):
    def __init__(self, p=4., eps=1e-6, learn_p=False,num_channels=22, epsilon=1e-5):
        super().__init__()
        self._p = p
        self._learn_p = learn_p
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.set_learn_p(flag=learn_p)
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon  # 防止除零的极小常数
        self.alpha_param = torch.nn.Parameter(torch.tensor(0.5))
    def set_learn_p(self, flag):
        self._learn_p = flag
        self.p.requires_grad = flag
    def forward(self, x):
        # 假设输入 x 的形状为 [batch_size, channels, time_steps]
        o = self.timeavg(x)
        embedding = (o + self.epsilon).pow(0.5) * self.alpha
        # 计算归一化因子基于方差的结果
        norm = (self.gamma) / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
        ).pow(0.5)
        gate = 1 + torch.tanh(embedding * norm + self.beta)
        return x * gate
    def timeavg(self,x):
        o =  x.clamp(min=self.eps).log().mean(dim=-1).exp().unsqueeze(2)
     #   o = (x.clamp(min=self.eps).pow(self.p).exp()).mean(dim=-1).log().pow(1.0 / self.p).unsqueeze(2)
#     o = F.lp_pool1d(x.clamp(min=self.eps), norm_type=self.p, kernel_size=x.size(-1))
        return o
class RMSPool1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
    def forward(self, x):
        # 计算局部窗口内的均方根值
        square_mean = F.avg_pool1d(x**2, self.kernel_size, self.stride, self.padding)
        rms = torch.sqrt(square_mean)
        return rms
class RMSfusion(nn.Module):
    def __init__(self, kernel_sizes=[5, 10, 25]):
        super().__init__()
        # 使用多个不同大小的核进行池化操作，这些核的大小在初始化时被指定
        self.RMS_layers = nn.ModuleList()  # 存储多个池化层
        self.L = len(kernel_sizes)  # 核的数量
        for k in kernel_sizes:
            # 为每个核大小创建一个池化层序列，包括自定义的 VarPool1D 和 Flatten 操作
            self.RMS_layers.append(
                nn.Sequential(
                    RMSPool1D(kernel_size=k, stride=int(k / 2)),  # 池化层
                    nn.Flatten(start_dim=1),  # 扁平化层，从第一个维度开始
                )
            )
    def forward(self, x,x1,x2):
        out = []
        x = self.RMS_layers[0](x)
        out.append(x)
        x1 = self.RMS_layers[1](x1)
        out.append(x1)
        x2 = self.RMS_layers[2](x2)
        out.append(x2)
        # 将处理后的结果在特征维度上拼接起来
        y = torch.concat(out, dim=1)
        return y  # 返回拼接后的结果
class EAEEG_Encoder(nn.Module):
    def __init__(
        self,
        middle_channel=47,
        multiple=9,
        pool_kernels=[50, 100, 250],
    ):
        super().__init__()
        self.pool = MultiScalePooling(in_channels=middle_channel)
        self.conv = Conv(in_channels=22, out_channels=multiple,kernel_size=75)
        self.conv1 = Conv1(in_channels=198, out_channels=middle_channel,kernel_size=1)
        self.GeMP1D =GeMP1D(num_channels=198)
        self.fftmix = RMSfusion(kernel_sizes=pool_kernels)
    def forward(self, x):
        # 数据预处理和归一化
        x = self.conv(x)
        out_mean = torch.mean(x, 2, True)
        out_var = torch.mean(x ** 2, 2, True)
        x = (x - out_mean) / torch.sqrt(out_var + 1e-5)
        x = self.GeMP1D(x)
        # 特征提取
        x = self.conv1(x)
        x,x1,x2 = self.pool(x)
        x = self.fftmix(x,x1,x2)
        return x
class Pooling1D(nn.Module):
    """
    PoolFormer池化操作的实现，已修改为适用于 [B, C, T] 输入（EEG信号）
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        # 使用1D池化操作处理时间维度
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)
    def forward(self, x):
        # 输入 x 的形状为 [B, C, T]
        y = self.pool(x)  # 沿时间维度进行池化
        return y - x  # 返回池化输出和输入之间的差值
class EAEEG(nn.Module):
    def __init__(
        self,
        chans,           # 输入信号的通道数（如EEG信号的电极数）
        samples,         # 输入信号的采样点数（如EEG每个通道的时间点数）
        multiple = 9,
        middle_channel = 47,
        num_classes=4,
        pool_kernels=[50, 100, 200],  # 池化层的核大小列表
    ):
        super().__init__()
        # 创建编码器（Efficient_Encoder），用于提取特征
        self.encoder = EAEEG_Encoder(
            middle_channel=middle_channel,
            multiple=multiple,
            pool_kernels=pool_kernels,
        )
        self.features = None  # 用于存储特征
        # 用一个形状为 (1, chans, samples) 的张量作为输入，通过编码器获取输出特征维度
        x = torch.ones((1, chans, samples))
        out = self.encoder(x)
        feat_dim = out.shape[-1]  # 输出特征的最后一个维度大小
        # 定义原型参数用于分类：
        # * Inter-class Separation Prototype (ISP)：类间分离原型
        self.isp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        # * Intra-class Compactness (ICP)：类内紧密性原型
        self.icp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        # 初始化 ISP 参数，使用 Kaiming 正态初始化
        nn.init.kaiming_normal_(self.isp)
    def get_features(self):
        # 如果 features 不为空，则返回已提取的特征
        if self.features is not None:
            return self.features
        else:
            # 否则抛出运行时错误，提示未运行 forward() 提取特征
            raise RuntimeError("No features available. Run forward() first.")
    def forward(self, x):
        # 通过编码器提取特征
        features = self.encoder(x)
        self.features = features  # 存储提取的特征
        # 对 isp 参数进行重新归一化，使每个类的原型向量的范数不超过1
        self.isp.data = torch.renorm(self.isp.data, p=2, dim=0, maxnorm=1)
        # 通过爱因斯坦求和约定进行特征与 ISP 原型之间的点积运算，得到分类的 logits
        logits = torch.einsum("bd,cd->bc", features, self.isp)
        return logits  # 返回分类结果
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # a simple test
    model = EAEEG(chans=22, samples=1000, num_classes=4,middle_channel = 47)
    model = model.to(device)
    inp = torch.rand(10, 22, 1000).to(device)
    out = model(inp).to(device)
    # Print the number of trainable parameters
    summary(model, (22, 1000))
    print(out.shape)
