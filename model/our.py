import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import model.resnet as models
import random

manual_seed = 321
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2,
                 zoom_factor=8, criterion=nn.CrossEntropyLoss(ignore_index=255),
                 BatchNorm=nn.BatchNorm2d, pretrained=True, args=None):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.classes = classes
        models.BatchNorm = BatchNorm
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        fea_dim = 2048
        self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins, BatchNorm)
        fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 512, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, 256, kernel_size=1)
            )
        main_dim = 512
        aux_dim = 256
        self.main_proto = nn.Parameter(torch.randn(self.classes, main_dim).cuda())
        self.aux_proto = nn.Parameter(torch.randn(self.classes, aux_dim).cuda())
        gamma_dim = 1
        self.gamma_conv = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, gamma_dim)
        )
        self.d = main_dim
        self.WQ = nn.Linear(self.d, self.d)
        self.WK = nn.Linear(self.d, self.d)
        self.WV = nn.Linear(self.d, self.d)
        self.linear_fusion = nn.Linear(2 * self.d, self.d)
        self.attn_conv = nn.Conv2d(6, 1, kernel_size=7, padding=3, bias=False)
        self.args = args

        self.attention_balance_net = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(512, 512 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 16, 512, bias=False)
        )

        self.modulate_weight = nn.Parameter(torch.ones(1) * 0.5).cuda()
        self.temperature_base = 0.1
        self.temperature_scale = nn.Parameter(torch.ones(1) * 0.05).cuda()

    def calculate_cross_class_similarity(self, base_proto, novel_proto):
        base_proto_norm = F.normalize(base_proto, p=2, dim=1)  # [M, d]
        novel_proto_norm = F.normalize(novel_proto, p=2, dim=1)  # [N, d]

        similarity_matrix = torch.matmul(novel_proto_norm, base_proto_norm.t())  # [N, M]

        max_similarity = similarity_matrix.max(dim=1, keepdim=True)[0]  # [N, 1]

        alpha = 1.0 - max_similarity  # [N, 1]

        alpha = torch.clamp(alpha, min=0.1, max=0.9)

        return alpha

    def calculate_prototype_statistics(self, prototypes):

        mu = prototypes.mean(dim=1, keepdim=True)  # [K, 1]

        sigma = prototypes.std(dim=1, unbiased=False, keepdim=True)  # [K, 1]

        avg_mu = mu.mean()  # 标量

        avg_sigma = sigma.mean()  # 标量

        return mu, sigma, avg_mu, avg_sigma
    # 新类自适应对齐模块
    def novel_classifier_calibration(self, base_proto, novel_proto):

        alpha = self.calculate_cross_class_similarity(base_proto, novel_proto)  # [N, 1]

        base_mu, base_sigma, avg_mu_b, avg_sigma_b = self.calculate_prototype_statistics(base_proto)

        novel_mu, novel_sigma, avg_mu_n, avg_sigma_n = self.calculate_prototype_statistics(novel_proto)

        avg_mu_b_exp = avg_mu_b.unsqueeze(0).unsqueeze(0)  # [1, 1]
        avg_mu_n_exp = avg_mu_n.unsqueeze(0).unsqueeze(0)  # [1, 1]

        calibrated_mean = novel_proto - avg_mu_n_exp + avg_mu_b_exp

        z_after_mean_shift = (1 - alpha) * novel_proto + alpha * calibrated_mean

        sigma_ratio = avg_sigma_b / (novel_sigma + 1e-12)  # [N, 1]

        adaptive_sigma_ratio = 1.0 + alpha * (sigma_ratio - 1.0)  # [N, 1]

        z_calibrated = z_after_mean_shift * adaptive_sigma_ratio

        return z_calibrated

    # 语义关联迁移模块
    def modulate_new_prototypes(self, base_proto, novel_proto):

        base_proto = F.normalize(base_proto.float(), p=2, dim=1)  # 强制归一化
        novel_proto = F.normalize(novel_proto.float(), p=2, dim=1)  # 强制归一化
        M = base_proto.size(0)
        N = novel_proto.size(0)
        modulated_protos = []

        temperature = 1

        for i in range(N):
            u_i = novel_proto[i]  # [d]
            Q = self.WQ(u_i).unsqueeze(0)  # [1, d]
            K = self.WK(base_proto)  # [M, d]
            V = self.WV(base_proto)  # [M, d]

            Q = F.normalize(Q, p=2, dim=-1)
            K = F.normalize(K, p=2, dim=-1)

            sim = torch.matmul(Q, K.t())  # [1, M]
            weight = F.softmax(sim / temperature, dim=1)  # [1, M]

            a_ui = torch.matmul(weight, V)  # [1, d]
            a_ui = F.normalize(a_ui, p=2, dim=-1)
            concat_vec = torch.cat([u_i.unsqueeze(0), a_ui], dim=1)  # [1, 2d]
            modulated_u = self.linear_fusion(concat_vec).squeeze(0)  # [d]
            modulated_u = F.normalize(modulated_u, p=2, dim=-1)

            modulate_weight = torch.sigmoid(self.modulate_weight)  # 映射到0~1
            modulated_u = modulate_weight * modulated_u + (1 - modulate_weight) * u_i
            modulated_u = F.normalize(modulated_u, p=2, dim=-1)

            modulated_protos.append(modulated_u)

        modulated_protos = torch.stack(modulated_protos, dim=0)
        return modulated_protos  # [N, d]

    def forward(self, x, y=None, gened_proto=None, base_num=11, novel_num=5, iter=None,
                gen_proto=False, eval_model=False, visualize=False):
        self.iter = iter
        self.base_num = base_num

        def WG(x, y, proto, target_cls):

            with torch.no_grad():
                b, c, h, w = x.size()[:]
                tmp_y = F.interpolate(y.float().unsqueeze(1), size=(h, w), mode='nearest')
                out = x.clone()
                unique_y = list(tmp_y.unique())
                new_gen_proto = proto.data.clone()
                # 多尺度空间-通道动态注意力模块
                def spatial_attention(feat):
                    avg_pool_global = torch.mean(feat, dim=1, keepdim=True)  # [b, 1, h, w]
                    max_pool_global = torch.max(feat, dim=1, keepdim=True)[0]  # [b, 1, h, w]
                    kernel_sizes = [3, 5]
                    local_feats = []
                    for k in kernel_sizes:
                        padding = k // 2
                        avg_pool_local = F.avg_pool2d(feat, kernel_size=k, stride=1, padding=padding)
                        avg_pool_local = torch.mean(avg_pool_local, dim=1, keepdim=True)  # [b, 1, h, w]
                        max_pool_local = F.max_pool2d(feat, kernel_size=k, stride=1, padding=padding)
                        max_pool_local = torch.max(max_pool_local, dim=1, keepdim=True)[0]  # [b, 1, h, w]
                        local_feats.extend([avg_pool_local, max_pool_local])
                    attn = torch.cat([avg_pool_global, max_pool_global] + local_feats, dim=1)  # [b, 6, h, w]
                    attn = self.attn_conv(attn)
                    spatial_attn = torch.sigmoid(attn)  # [b, 1, h, w]
                    return spatial_attn

                def channel_attention(feat):
                    avg_pool = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)  # [b, c]
                    max_pool = F.adaptive_max_pool2d(feat, 1).squeeze(-1).squeeze(-1)  # [b, c]
                    avg_out = self.channel_mlp(avg_pool)
                    max_out = self.channel_mlp(max_pool)
                    channel_attn = torch.sigmoid(avg_out + max_out)  # [b, c]
                    return channel_attn.unsqueeze(-1).unsqueeze(-1)  # [b, c, 1, 1]

                def compute_balance_weights(feat):
                    B, C, H, W = feat.shape

                    channel_means = feat.mean(dim=[2, 3])  # [B, C] 通道空间均值
                    channel_mean_mean = channel_means.mean(dim=1)  # [B] 通道均值的均值
                    channel_mean_std = channel_means.std(dim=1, unbiased=False)  # [B] 通道均值的标准差（方差）

                    channel_stds = feat.std(dim=[2, 3], unbiased=False)  # [B, C] 通道空间标准差
                    channel_std_mean = channel_stds.mean(dim=1)  # [B] 通道标准差的均值
                    channel_std_std = channel_stds.std(dim=1, unbiased=False)  # [B] 通道标准差的标准差（方差）

                    spatial_means = feat.mean(dim=1)  # [B, H, W] 空间均值（通道平均）
                    spatial_mean_mean = spatial_means.mean(dim=[1, 2])  # [B] 空间均值的均值
                    spatial_mean_std = spatial_means.std(dim=[1, 2], unbiased=False)  # [B] 空间均值的标准差（方差）

                    spatial_stds = feat.std(dim=1, unbiased=False)  # [B, H, W] 空间标准差（通道平均）
                    spatial_std_mean = spatial_stds.mean(dim=[1, 2])  # [B] 空间标准差的均值
                    spatial_std_std = spatial_stds.std(dim=[1, 2], unbiased=False)  # [B] 空间标准差的标准差（方差）

                    stats = torch.stack([
                        channel_mean_mean,  # μ_cm
                        channel_mean_std,  # σ_cm
                        channel_std_mean,  # μ_σc
                        channel_std_std,  # σ_σc
                        spatial_mean_mean,  # μ_sm
                        spatial_mean_std,  # σ_sm
                        spatial_std_mean,  # μ_s
                        spatial_std_std  # σ_s
                    ], dim=1)  # [B, 8]

                    balance_weights = self.attention_balance_net(stats)  # [B, 2]
                    balance_weights = F.softmax(balance_weights, dim=1)

                    spatial_weight = balance_weights[:, 0].view(B, 1, 1, 1)
                    channel_weight = balance_weights[:, 1].view(B, 1, 1, 1)
                    return spatial_weight, channel_weight

                spatial_attn = spatial_attention(out)  # [b, 1, h, w]
                channel_attn = channel_attention(out)  # [b, c, 1, 1]

                spatial_weight, channel_weight = compute_balance_weights(out)

                spatial_attn_expanded = spatial_attn.expand_as(out)  # [b, c, h, w]
                channel_attn_expanded = channel_attn.expand_as(out)  # [b, c, h, w]
                combined_attn = spatial_weight * spatial_attn_expanded + channel_weight * channel_attn_expanded
                weighted_out = out * combined_attn  # [b, c, h, w]


                for tmp_cls in unique_y:
                    if tmp_cls == 255:
                        continue
                    tmp_mask = (tmp_y.float() == tmp_cls.float()).float()  # [b, 1, h, w]
                    weighted_sum = (weighted_out * tmp_mask).sum(0).sum(-1).sum(-1)  # [c]
                    mask_sum = tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12
                    tmp_p = weighted_sum / mask_sum  # [c]
                    new_gen_proto[tmp_cls.long(), :] = tmp_p
                return new_gen_proto


        def generate_fake_proto(proto, x, y): #训练
            b, c, h, w = x.size()[:]
            tmp_y = F.interpolate(y.float().unsqueeze(1), size=(h, w), mode='nearest')
            unique_y = list(tmp_y.unique())
            raw_unique_y = list(tmp_y.unique())
            if 0 in unique_y:
                unique_y.remove(0)
            if 255 in unique_y:
                unique_y.remove(255)
            novel_num = len(unique_y) // 2
            fake_novel = random.sample(unique_y, novel_num)
            for fn in fake_novel:
                unique_y.remove(fn)
            fake_context = unique_y
            new_proto = self.main_proto.clone()
            new_proto = new_proto / (torch.norm(new_proto, 2, 1, True) + 1e-12)
            x = x / (torch.norm(x, 2, 1, True) + 1e-12)

            def spatial_attention(feat):
                # feat: [b, c, h, w]

                avg_pool_global = torch.mean(feat, dim=1, keepdim=True)  # [b, 1, h, w]
                max_pool_global = torch.max(feat, dim=1, keepdim=True)[0]  # [b, 1, h, w]

                kernel_sizes = [3, 5]
                local_feats = []
                for k in kernel_sizes:
                    padding = k // 2
                    avg_pool_local = F.avg_pool2d(feat, kernel_size=k, stride=1, padding=padding)  # [b, c, h, w]
                    avg_pool_local = torch.mean(avg_pool_local, dim=1, keepdim=True)  # [b, 1, h, w]
                    max_pool_local = F.max_pool2d(feat, kernel_size=k, stride=1, padding=padding)  # [b, c, h, w]
                    max_pool_local = torch.max(max_pool_local, dim=1, keepdim=True)[0]  # [b, 1, h, w]
                    local_feats.extend([avg_pool_local, max_pool_local])

                attn = torch.cat([avg_pool_global, max_pool_global] + local_feats,
                                 dim=1)
                attn = self.attn_conv(attn)  # [b, 1, h, w]
                return torch.sigmoid(attn)  # [b, 1, h, w]

            def channel_attention(feat):
                # feat: [b, c, h, w]
                avg_pool = F.adaptive_avg_pool2d(feat, 1)  # [b, c, 1, 1]
                max_pool = F.adaptive_max_pool2d(feat, 1)  # [b, c, 1, 1]
                avg_out = self.channel_mlp(avg_pool.squeeze(-1).squeeze(-1))  # [b, c]
                max_out = self.channel_mlp(max_pool.squeeze(-1).squeeze(-1))  # [b, c]
                channel_attn = torch.sigmoid(avg_out + max_out)  # [b, c]
                return channel_attn.unsqueeze(-1).unsqueeze(-1)  # [b, c, 1, 1]

            spatial_attn = spatial_attention(x)  # [b, 1, h, w]
            channel_attn = channel_attention(x)  # [b, c, 1, 1]
            spatial_attn_expanded = spatial_attn.expand_as(x)  # [b, c, h, w]
            channel_attn_expanded = channel_attn.expand_as(x)  # [b, c, h, w]

            def compute_balance_weights(feat):
                B, C, H, W = feat.shape

                # 1. 通道维度统计
                channel_means = feat.mean(dim=[2, 3])  # [B, C] 通道空间均值
                channel_mean_mean = channel_means.mean(dim=1)  # [B] 通道均值的均值
                channel_mean_var = channel_means.var(dim=1)  # [B] 通道均值的标准差（方差）

                channel_stds = feat.std(dim=[2, 3])  # [B, C] 通道空间标准差
                channel_std_mean = channel_stds.mean(dim=1)  # [B] 通道标准差的均值
                channel_std_var = channel_stds.var(dim=1)  # [B] 通道标准差的标准差（方差）

                # 2. 空间维度统计
                spatial_means = feat.mean(dim=1)  # [B, H, W] 空间均值（通道平均）
                spatial_mean_mean = spatial_means.mean(dim=[1, 2])  # [B] 空间均值的均值
                spatial_mean_var = spatial_means.var(dim=[1, 2])  # [B] 空间均值的标准差（方差）

                spatial_stds = feat.std(dim=1)  # [B, H, W] 空间标准差（通道平均）
                spatial_std_mean = spatial_stds.mean(dim=[1, 2])  # [B] 空间标准差的均值
                spatial_std_var = spatial_stds.var(dim=[1, 2])  # [B] 空间标准差的标准差（方差）

                stats = torch.stack([
                    channel_mean_mean,  # μ_cm
                    channel_mean_var,  # σ_cm
                    channel_std_mean,  # μ_σc
                    channel_std_var,  # σ_σc
                    spatial_mean_mean,  # μ_sm
                    spatial_mean_var,  # σ_sm
                    spatial_std_mean,  # μ_s
                    spatial_std_var  # σ_s
                ], dim=1)  # [B, 8]

                # 4. 计算平衡权重
                balance_weights = self.attention_balance_net(stats)  # [B, 2]
                balance_weights = F.softmax(balance_weights, dim=1)

                spatial_weight = balance_weights[:, 0].view(B, 1, 1, 1)
                channel_weight = balance_weights[:, 1].view(B, 1, 1, 1)
                return spatial_weight, channel_weight

            spatial_weight, channel_weight = compute_balance_weights(x)
            combined_attn = spatial_weight * spatial_attn_expanded + channel_weight * channel_attn_expanded  # [b, c, h, w]
            weighted_x = x * combined_attn  # [b, c, h, w]

            fake_novel_protos = {}
            for fn in fake_novel:
                tmp_mask = (tmp_y == fn).float()  # [b, 1, h, w]
                tmp_feat = (weighted_x * tmp_mask).sum(dim=[2, 3]) / (tmp_mask.sum(dim=[2, 3]) + 1e-12)  # [b, c]
                tmp_feat = tmp_feat.mean(dim=0)  # [c]
                fake_novel_protos[fn] = tmp_feat
                fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
                fake_vec[fn.long()] = 1
                new_proto = new_proto * (1 - fake_vec) + tmp_feat.unsqueeze(0) * fake_vec

            fake_context_protos = {}
            for fc in fake_context:
                tmp_mask = (tmp_y == fc).float()
                tmp_feat = (weighted_x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)
                fake_context_protos[fc] = tmp_feat
                fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
                fake_vec[fc.long()] = 1
                raw_feat = new_proto[fc.long()].clone()
                all_feat = torch.cat([raw_feat, tmp_feat], 0).unsqueeze(0)
                ratio = F.sigmoid(self.gamma_conv(all_feat))[0]
                new_proto = new_proto * (1 - fake_vec) + (
                        (raw_feat * ratio + tmp_feat * (1 - ratio)).unsqueeze(0) * fake_vec)

            context_proto_matrix = torch.stack([fake_context_protos[fc] for fc in fake_context], dim=0)  # [M, d]
            for fn in fake_novel:
                novel_proto = fake_novel_protos[fn].unsqueeze(0)  # [1, d]
                novel_proto = F.normalize(novel_proto, p=2, dim=-1)
                context_proto_matrix = F.normalize(context_proto_matrix, p=2, dim=-1)
                modulated_novel_proto = self.modulate_new_prototypes(
                    base_proto=context_proto_matrix,
                    novel_proto=novel_proto
                ).squeeze(0)  # [d]
                fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
                fake_vec[fn.long()] = 1
                new_proto = new_proto * (1 - fake_vec) + modulated_novel_proto.unsqueeze(0) * fake_vec
                new_proto = F.normalize(new_proto, p=2, dim=-1)

            if random.random() > 0.5 and 0 in raw_unique_y:
                tmp_mask = (tmp_y == 0).float()  # [b, 1, h, w]
                weighted_mask = tmp_mask * combined_attn.mean(dim=1, keepdim=True)  # [b, 1, h, w]
                tmp_feat = (x * weighted_mask).sum(dim=[2, 3]) / (weighted_mask.sum(dim=[2, 3]) + 1e-12)  # [b, c]
                tmp_feat = tmp_feat.mean(dim=0)  # [c]
                fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
                fake_vec[0] = 1
                raw_feat = new_proto[0].clone()
                all_feat = torch.cat([raw_feat, tmp_feat], 0).unsqueeze((0))  # 1, 1024
                ratio = F.sigmoid(self.gamma_conv(all_feat))[0]  # 512
                new_proto = new_proto * (1 - fake_vec) + (
                        (raw_feat * ratio + tmp_feat * (1 - ratio)).unsqueeze(0) * fake_vec)
                new_proto = F.normalize(new_proto, p=2, dim=-1)

            replace_proto = new_proto.clone()
            return new_proto, replace_proto

        if gen_proto: # 新类注册
            # proto generation
            # supp_x: [cls, s, c, h, w]
            # supp_y: [cls, s, h, w]
            x = x[0]
            y = y[0]
            cls_num = x.size(0)
            shot_num = x.size(1)
            with torch.no_grad():
                gened_proto = self.main_proto.clone()
                base_proto_list = []
                tmp_x_feat_list = []
                tmp_gened_proto_list = []
                for idx in range(cls_num):
                    tmp_x = x[idx]
                    tmp_y = y[idx]
                    raw_tmp_y = tmp_y
                    tmp_x = self.layer0(tmp_x)
                    tmp_x = self.layer1(tmp_x)
                    tmp_x = self.layer2(tmp_x)
                    tmp_x = self.layer3(tmp_x)
                    tmp_x = self.layer4(tmp_x)
                    layer4_x = tmp_x
                    tmp_x = self.ppm(tmp_x)
                    ppm_feat = tmp_x.clone()
                    tmp_x = self.cls(tmp_x)
                    tmp_x_feat_list.append(tmp_x)
                    tmp_cls = idx + base_num
                    tmp_gened_proto = WG(x=tmp_x, y=tmp_y, proto=self.main_proto, target_cls=tmp_cls)
                    tmp_gened_proto_list.append(tmp_gened_proto)
                    base_proto_list.append(tmp_gened_proto[:base_num, :].unsqueeze(0))
                    gened_proto[tmp_cls, :] = tmp_gened_proto[tmp_cls, :]
                base_proto = torch.cat(base_proto_list, 0).mean(0)
                base_proto = F.normalize(base_proto, p=2, dim=-1)
                novel_proto_original = gened_proto[base_num:, :].clone()
                novel_proto_original = F.normalize(novel_proto_original, p=2, dim=-1)

                modulated_novel_proto = self.modulate_new_prototypes(base_proto, novel_proto_original)

                novel_base_sim = torch.matmul(novel_proto_original, base_proto.t())  # [N, M]
                novel_base_avg_sim = novel_base_sim.mean(dim=1, keepdim=True)  # [N, 1]

                alpha = 0.6
                beta = 0.2
                modulate_ratio = novel_base_avg_sim * alpha + beta
                modulate_ratio = torch.clamp(modulate_ratio, min=0.1, max=0.7)

                fused_novel_proto = modulate_ratio * modulated_novel_proto + (1 - modulate_ratio) * novel_proto_original
                fused_novel_proto = F.normalize(fused_novel_proto, p=2, dim=-1)

                gened_proto[base_num:, :] = fused_novel_proto
                gened_proto = F.normalize(gened_proto, p=2, dim=-1)


                ori_proto = F.normalize(self.main_proto[:base_num, :], p=2, dim=-1)
                all_proto = torch.cat([ori_proto, base_proto], 1)
                ratio = F.sigmoid(self.gamma_conv(all_proto))  # n, 512
                base_proto = ratio * ori_proto + (1 - ratio) * base_proto
                base_proto = F.normalize(base_proto, p=2, dim=-1)

                novel_proto = gened_proto[base_num:, :].clone()
                novel_proto = F.normalize(novel_proto, p=2, dim=-1)

                calibrated_novel_proto = self.novel_classifier_calibration(base_proto, novel_proto)
                calibrated_novel_proto = F.normalize(calibrated_novel_proto, p=2, dim=-1)

                gened_proto = torch.cat([base_proto, calibrated_novel_proto], 0)
                gened_proto = F.normalize(gened_proto, p=2, dim=-1)
            return gened_proto.unsqueeze(0)
        else:
            x_size = x.size()
            assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
            h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
            w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x_tmp = self.layer3(x)
            x = self.layer4(x_tmp)
            x = self.ppm(x)
            x = self.cls(x)
            raw_x = x.clone()
            if eval_model:
                #### evaluation
                if len(gened_proto.size()[:]) == 3:
                    gened_proto = gened_proto[0]
                if visualize:
                    vis_feat = x.clone()
                refine_proto = self.post_refine_proto_v2(proto=self.main_proto, x=raw_x)
                refine_proto[:, :base_num] = refine_proto[:, :base_num] + gened_proto[:base_num].unsqueeze(0)
                refine_proto[:, base_num:] = refine_proto[:, base_num:] * 0 + gened_proto[base_num:].unsqueeze(0)
                x = self.get_pred(raw_x, refine_proto)
            else:
                ##### training
                fake_num = x.size(0) // 2
                ori_new_proto, replace_proto = generate_fake_proto(proto=self.main_proto, x=x[fake_num:],
                                                                   y=y[fake_num:])
                x = self.get_pred(x, ori_new_proto)
                x_pre = x.clone()
                refine_proto = self.post_refine_proto_v2(proto=self.main_proto, x=raw_x)
                post_refine_proto = refine_proto.clone()
                post_refine_proto[:, :base_num] = post_refine_proto[:, :base_num] + ori_new_proto[:base_num].unsqueeze(
                    0)
                post_refine_proto[:, base_num:] = post_refine_proto[:, base_num:] * 0 + ori_new_proto[
                                                                                        base_num:].unsqueeze(0)
                x = self.get_pred(raw_x, post_refine_proto)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            if self.training:
                aux = self.aux(x_tmp)
                aux = self.get_pred(aux, self.aux_proto)
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                main_loss = self.criterion(x, y)
                aux_loss = self.criterion(aux, y)
                x_pre = F.interpolate(x_pre, size=(h, w), mode='bilinear', align_corners=True)
                pre_loss = self.criterion(x_pre, y)
                main_loss = 0.5 * main_loss + 0.5 * pre_loss
                return x.max(1)[1], main_loss, aux_loss
            else:
                if visualize:
                    return x, vis_feat
                else:
                    return x

    def post_refine_proto_v2(self, proto, x):
        raw_x = x.clone()
        b, c, h, w = raw_x.shape[:]
        pred = self.get_pred(x, proto).view(b, proto.shape[0], h * w)
        pred = F.softmax(pred, 2)
        pred_proto = pred @ raw_x.view(b, c, h * w).permute(0, 2, 1)
        pred_proto_norm = F.normalize(pred_proto, 2, -1)  # b, n, c
        proto_norm = F.normalize(proto, 2, -1).unsqueeze(0)  # 1, n, c
        pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)  # b, n, 1
        pred_weight = pred_weight * (pred_weight > 0).float()
        pred_proto = pred_weight * pred_proto + (1 - pred_weight) * proto.unsqueeze(0)  # b, n, c
        return pred_proto

    def get_pred(self, x, proto):
        b, c, h, w = x.size()[:]
        if len(proto.shape[:]) == 3:
            # x: [b, c, h, w]
            # proto: [b, cls, c]
            cls_num = proto.size(1)
            x = F.normalize(x, p=2, dim=1)
            proto = F.normalize(proto, p=2, dim=-1)  # b, n, c
            x = x.contiguous().view(b, c, h * w)  # b, c, hw
            pred = proto @ x  # b, cls, hw
        elif len(proto.shape[:]) == 2:
            cls_num = proto.size(0)
            x = F.normalize(x, p=2, dim=1) 
            proto = F.normalize(proto, p=2, dim=-1)
            x = x.contiguous().view(b, c, h * w)  # b, c, hw
            proto = proto.unsqueeze(0)  # 1, cls, c
            pred = proto @ x  # b, cls, hw
        pred = pred.contiguous().view(b, cls_num, h, w)
        return pred * 10
