import torch
import torch.nn as nn
import math


class HistogramSpacialNet(nn.Module):
    def __init__(self, num_seq = 2, scale = [1,2,5], nm_flag = False, bins_num=128):
        super(HistogramSpacialNet, self).__init__()
        cat_num = 1
        if nm_flag:
            cat_num += 1
        self.bins_num = bins_num
        self.features = nn.Sequential( 
            nn.Conv1d((sum([i**2 for i in scale]) + cat_num)*num_seq, 64, kernel_size=4, stride=2, padding=1), # [64, 128] -> [64, 64]
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), #[128, 32] 
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=4, padding=0), # [256, 8]
            nn.ReLU(inplace=True) 
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8*256, 512),
        )
        self._initialize_weights()

    def forward(self, x):
        f = self.features(x)
        x = f.view(f.size(0), 8*256) 
        x = self.classifier(x)
        return f, x

    def _initialize_weights(self):
        cnt = 0
        for m in self.modules():
            
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                cnt += 1
                if cnt == 3:
                    m.weight.data.normal_(0, 0.05)
                else:
                    m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def histogramspacialnet(bins_num = 128, nm_flag = False):
    model = HistogramSpacialNet(bins_num=bins_num, nm_flag = nm_flag)
    return model

class l2norm(nn.Module):
    def __init__(self):
        super(l2norm,self).__init__()

    def forward(self,input,epsilon = 1e-7):
        assert len(input.size()) == 2,"Input dimension requires 2,but get {}".format(len(input.size()))
        
        norm = torch.norm(input,p = 2,dim = 1,keepdim = True)
        output = torch.div(input,norm+epsilon)
        return output

class scaleTime(nn.Module):
    def __init__(self):
        super(scaleTime,self).__init__()
        self.Tmax = torch.FloatTensor([12])
        self.Tmin = torch.FloatTensor([-8])
        if torch.cuda.is_available():
            self.Tmax = self.Tmax.cuda()
            self.Tmin = self.Tmin.cuda()
            

    def forward(self, input):
        input = torch.sigmoid(input)
        output = input * (self.Tmax - self.Tmin) + self.Tmin
        # output = torch.log2(torch.exp(2 * (input - 0.5) * torch.log(self.Tmax)))
        return output

class ExposureStrategyNet(nn.Module):
    def __init__(self, hidden_dim = 512, bins_num = 128, joint_learning_flag = True, nm_flag = False):
        super(ExposureStrategyNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.joint_learning_flag = joint_learning_flag
        self.histogram = histogramspacialnet(bins_num = bins_num, nm_flag = nm_flag)
        self.l2norm = l2norm()
        self.gru = nn.GRUCell(input_size = hidden_dim, hidden_size = hidden_dim) 
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        if  self.joint_learning_flag:
            # self.hist_regression = nn.Sequential(
            self.hdr = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.Softmax(dim=1)
            )
        self.scaleTime = scaleTime() 
        self._initialize_weights()
        
        
    def forward(self, hist, h = None):
        _, fhis = self.histogram(hist)
        if h is None:
            feature = fhis
        else:
            h = self.gru(fhis, h)
            feature = h
            
        output = self.classifier(feature)
        if self.joint_learning_flag:
            hist_output = self.hdr(feature)
        else:
            hist_output = None
            
        EV_output = output = self.scaleTime(output)        
        return EV_output, hist_output, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(batch_size, self.hidden_dim).zero_()
        return hidden
    
    def _initialize_weights(self):
        cnt = 0
        for m in self.classifier.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                cnt += 1
                if cnt == 2:
                    m.weight.data.normal_(0, 0.05)
                else:
                    m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.joint_learning_flag:
            for m in self.hdr.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2./n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    cnt += 1
                    if cnt == 2:
                        m.weight.data.normal_(0, 0.05)
                    else:
                        m.weight.data.normal_(0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()