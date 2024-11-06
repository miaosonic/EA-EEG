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
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, groups=1, bias=True):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, out_channels, kernel_size, dilation=dilation, padding=self.padding, groups=groups,
                              bias=bias)
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
        x = rearrange(x, "b (h c) t ->(b c) h t", h=1)
        # 卷积 -> 批归一化 -> 激活
        out = self.act(self.norm(self.conv(x)))
        # 调整输出形状
        out = rearrange(out, "(b c) h t ->b (h c) t", b=B)
        return out
class normalize_eeg_zscore(nn.Module):
    """
    对EEG信号进行Z-score归一化
    输入: eeg_data, 形状为 [B, C, T]
    """
    def __init__(self, ):
        super(normalize_eeg_zscore, self).__init__()
    def forward(self, eeg_data):
        mean = eeg_data.mean(dim=-1, keepdim=True)  # 计算每个通道的均值
        std = eeg_data.std(dim=-1, keepdim=True)  # 计算每个通道的标准差
        eeg_data_norm = (eeg_data - mean) / (std + 1e-6)  # 进行Z-score归一化，避免除以0
        return eeg_data_norm
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
        concatenated = torch.cat(pooled_outputs, dim=1)
        output = concatenated
        return output
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
class SequencePooling(nn.Module):
    def __init__(self, in_features):
        super(SequencePooling, self).__init__()
        self.attention = nn.Linear(in_features, out_features=1)
        self.apply(self.init_weight)
    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        attention_weights = torch.transpose(attention_weights, 1, 2)
        weighted_representation = torch.matmul(attention_weights, x)
        return torch.squeeze(weighted_representation, dim=-2)
    @staticmethod
    def init_weight(m):
      if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.1)
        nn.init.constant_(m.bias,0)
class SMixer1D(nn.Module):
    def __init__(self, dim, kernel_sizes=[5, 10, 25]):
        super().__init__()
        # 使用多个不同大小的核进行池化操作，这些核的大小在初始化时被指定
        self.var_layers = nn.ModuleList()  # 存储多个池化层
        self.L = len(kernel_sizes)  # 核的数量
        for k in kernel_sizes:
            # 为每个核大小创建一个池化层序列，包括自定义的 VarPool1D 和 Flatten 操作
            self.var_layers.append(
                nn.Sequential(
                    RMSPool1D(kernel_size=k, stride=int(k / 2)),  # 池化层
                    nn.Flatten(start_dim=1),  # 扁平化层，从第一个维度开始
                )
            )
    def forward(self, x):
        B, d, L = x.shape  # 获取输入的批量大小(B)、特征维度(d)、长度(L)
        # 将输入沿着特征维度 `d` 分成 `L` 份，每一份的大小为 d // self.L
        x_split = torch.split(x, d // self.L, dim=1)
        out = []  # 存储每个核处理后的结果
        for i in range(len(x_split)):
            # 使用对应的池化层对分割后的输入进行处理
            x = self.var_layers[i](x_split[i])
            out.append(x)
        # 将处理后的结果在特征维度上拼接起来
        y = torch.concat(out, dim=1)
        return y  # 返回拼接后的结果
class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []
        for i in range(out_shape):
            index = i*self.stride
            input = x[:, :, index:index+self.kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)
        out = torch.cat(out, dim=-1)
        return out
class Efficient_Encoder(nn.Module):
    def __init__(
        self,
        samples,
        chans,
        F1=16,
        F2=36,
        time_kernel1=75,
        pool_kernels=[50, 100, 250],
    ):
        super().__init__()
        self.pool = MultiScalePooling(in_channels=48)
        self.conv = Conv(in_channels=22, out_channels=9,kernel_size=75)
        self.conv3 = Conv(in_channels=48,out_channels=3,kernel_size=1)
        self.conv1 = Conv1(in_channels=198, out_channels=48,kernel_size=1)
        self.norm = normalize_eeg_zscore()
        self.GeMP1D =GeMP1D(num_channels=198)
        self.belt = nn.Parameter(torch.ones(1, 198, 1))
        self.fftmix = SMixer1D(dim=F2, kernel_sizes=pool_kernels)
        self.lazynorm = LayerNorm(features=(1, 198, 1))
        self.sp = SequencePooling(in_features=500)
        self.max =  nn.AvgPool1d(kernel_size=2, stride=2)
        self.var = VarPoold(kernel_size=1000,stride=1)
    def forward(self, x):
        # 数据预处理和归一化
        x = self.norm(x) # 使用 x0 作为权重调节 x
        x = self.conv(x)
        out_mean = torch.mean(x, 2, True)
        out_var = torch.mean(x ** 2, 2, True)
        x = (x - out_mean) / torch.sqrt(out_var + 1e-5)
        x = self.GeMP1D(x)
        # 特征提取
        x = self.lazynorm(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fftmix(x)
        return x
class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
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
        num_classes=4,   # 类别数量（分类任务的输出类别数）
        F1=9,            # 第一个卷积层的过滤器数量
        F2=48,           # 第二个卷积层的过滤器数量
        time_kernel1=75, # 时间卷积核的大小
        pool_kernels=[50, 100, 200],  # 池化层的核大小列表
    ):
        super().__init__()
        # 创建编码器（Efficient_Encoder），用于提取特征
        self.encoder = Efficient_Encoder(
            samples=samples,
            chans=chans,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
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
    model = EAEEG(chans=22, samples=1000, num_classes=4)
    model = model.to(device)
    inp = torch.rand(10, 22, 1000).to(device)
    out = model(inp).to(device)
    # Print the number of trainable parameters
    summary(model, (22, 1000))
    print(out.shape)
