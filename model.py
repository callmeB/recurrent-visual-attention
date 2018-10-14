import torch
import torch.nn as nn

from torch.distributions import Normal

from modules import baseline_network
from modules import cnn_network, core_network
from modules import action_network, location_network

import torch.nn.functional as F


class RecurrentAttention(nn.Module):
    def __init__(self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes, is_sampling=True):
        super(RecurrentAttention, self).__init__()
        self.std = std
        self.is_sampling = is_sampling

        # context
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 16)),
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True)
        )
        # emission
        self.emission = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
            nn.Tanh()
        )

        # glipse extractor
        self.retina = retina(g, k, s)

        # convolutional layers
        self.glimpse_conv = nn.Sequential(
            nn.Conv2d(3 * k, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # glimpse fc layer
        self.glimpse_fc = nn.Sequential(
            nn.Linear(g[0] * g[1] * 128, 1024)
        )
        # location layer
        self.glimpse_loc = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024)
        )

        # RNNs
        self.rnn_glimpses = nn.GRUCell(1024, 1024)
        self.rnn_locations = nn.GRUCell(1024, 1024)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2)
        )

        # baseline
        self.baseline = nn.Linear(2048, 1)


        #self.cnn1 = cnn_network(h_g, h_l, g, k, s, c)
        #self.cnn2 = cnn_network(h_g, h_l, g, k, s, c)
        #self.rnn = core_network(1024 * 2 * 2, hidden_size) # features * loc * nets
        #self.locator = location_network(hidden_size, 2, std)
        #self.classifier = action_network(hidden_size, num_classes)
        #self.baseliner = baseline_network(hidden_size, 1)

    def forward(self, x_a, x_b, num_glimpses):
        B, C, H, W = x_a.shape

        h_t_a = {"features": torch.zeros(B, 1024),
                 "locations": torch.zeros(B, 1024)}
        h_t_b = {"features": torch.zeros(B, 1024),
                 "locations": torch.zeros(B, 1024)}

        locs = []
        locs_mu = []

        log_pis = []
        baselines = []

        context_a = self.context(x_a)
        context_b = self.context(x_b)
        loc_a = self.emission(context_a.view(context_a.size()[0], -1))
        loc_b = self.emission(context_b.view(context_b.size()[0], -1))

        for i in range(num_glimpses):

            glimpse_a = self.retina.foveate(x_a, loc_a)
            glimpse_b = self.retina.foveate(x_b, loc_b)

            features_a = self.glimpse_conv(glimpse_a)
            features_b = self.glimpse_conv(glimpse_b)
            features_a = self.glimpse_fc(features_a.view(features_a.size()[0], -1))
            features_b = self.glimpse_fc(features_b.view(features_b.size()[0], -1))

            location_a = self.glimpse_loc(loc_a)
            location_b = self.glimpse_loc(loc_b)

            glimpse_a = torch.mul(features_a, location_a)
            glimpse_b = torch.mul(features_b, location_b)

            h_t_a["features"] = self.rnn_glimpses(glimpse_a, h_t_a["features"])
            h_t_a["locations"] = self.rnn_locations(h_t_a["features"], h_t_a["locations"])

            h_t_b["features"] = self.rnn_glimpses(glimpse_b, h_t_b["features"])
            h_t_b["locations"] = self.rnn_locations(h_t_b["features"], h_t_b["locations"])

            baseline = self.baseline(torch.cat((h_t_a["locations"].detach(), h_t_b["locations"].detach()), dim=1)).squeeze()
            baselines.append(baseline)

            mu_a = self.emission(h_t_a["locations"].detach())
            mu_b = self.emission(h_t_b["locations"].detach())

            mu = torch.cat((mu_a, mu_b), dim=1)

            if self.is_sampling:
                noise = torch.empty_like(mu)
                loc = mu + noise.normal_(std=self.std)
                loc = torch.tanh(loc)

                # we assume both dimensions are independent
                # 1. pdf of the joint is the product of the pdfs
                # 2. log of the product is the sum of the logs
                log_pi = Normal(mu, self.std).log_prob(loc)
                log_pis.append(torch.sum(log_pi, dim=1))
            else:
                loc = mu

            locs.append(loc)
            locs_mu.append(mu)

            loc_a, loc_b = torch.chunk(loc, 2, dim=1)

        features = torch.cat((h_t_a["features"], h_t_b["features"]), dim=1)
        log_probas = F.log_softmax(self.classifier(features), dim=1)

        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pis = torch.stack(log_pis).transpose(1, 0)

        return log_probas, log_pis, baselines, locs, locs_mu





        #g_t_1 = self.cnn1(x_a, x_b, l_t_a_prev, l_t_b1_prev)
        #g_t_2 = self.cnn2(x_a, x_b, l_t_a_prev, l_t_b2_prev)
        #g_t = torch.cat((g_t_1, g_t_2), dim=1)
        #h_t = self.rnn(g_t, h_t_prev)
        #mu_a, l_t_a, mu_b1, l_t_b1, mu_b2, l_t_b2 = self.locator(h_t[1])
        #b_t = self.baseliner(h_t).squeeze()

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        #log_pi_a = Normal(mu_a, self.std).log_prob(l_t_a)
        #log_pi_b1 = Normal(mu_b1, self.std).log_prob(l_t_b1)
        #log_pi_b2 = Normal(mu_b2, self.std).log_prob(l_t_b2)
        #log_pi = torch.cat((log_pi_a, log_pi_b1, log_pi_b2), dim=1)
        #log_pi = torch.sum(log_pi, dim=1)

        #if last:
        #    log_probas = self.classifier(h_t[0])
        #    return h_t, l_t_a, l_t_b1, l_t_b2, b_t, log_probas, log_pi
        #return h_t, l_t_a, l_t_b1, l_t_b2, b_t, log_pi


class retina(object):
    """
    A retina that extracts a foveated glimpse `phi`
    around location `l` from an image `x`. It encodes
    the region around `l` at a high-resolution but uses
    a progressively lower resolution for pixels further
    from `l`, resulting in a compressed representation
    of the original image `x`.

    Args
    ----
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l: a 2D Tensor of shape (B, 2). Contains normalized
      coordinates in the range [-1, 1].
    - g: size of the first square patch.
    - k: number of patches to extract in the glimpse.
    - s: scaling factor that controls the size of
      successive patches.

    Returns
    -------
    - phi: a 5D tensor of shape (B, k, g, g, C). The
      foveated glimpse of the image.
    """
    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        """
        Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a rectangle of
        size `g`, and each subsequent patch is a rectangle
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (h, w) and
        concatenated into a tensor of shape (B, k, C, h, w).
        """
        phi = []
        size = self.g

        # extract k patches of increasing size
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = self.s * size

        # resize the patches to rectangles of size g
        for i in range(1, len(phi)):
            h = phi[i].shape[-2] // self.g[0]
            w = phi[i].shape[-1] // self.g[1]
            phi[i] = F.avg_pool2d(phi[i], (h, w))

        # concatenate into a single tensor
        phi = torch.cat(phi, dim=1)
        return phi

    def extract_patch(self, x, l, patch_size):
        """
        Extract a single patch for each image in the
        minibatch `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, C, H, W). The minibatch
          of images.
        - l: a 2D Tensor of shape (B, 2).
        - size: a scalar defining the size of the extracted patch.

        Returns
        -------
        - patch: a 4D Tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        image_size = torch.tensor((H, W), dtype=torch.long)

        # denormalize coords of patch center
        coords = self.denormalize(l, image_size)

        # compute top left corner of patch
        patch = coords - (patch_size // 2)

        # loop through mini-batch and extract
        patches = []
        for i in range(B):
            image = x[i].unsqueeze(dim=0)

            # compute slice indices
            from_, to = patch[i], patch[i] + patch_size

            # pad tensor in case exceeds
            if self.exceeds(from_, to, image_size):
                pad_dims = (
                    patch_size[1]//2, patch_size[1]//2,
                    patch_size[0]//2, patch_size[0]//2,
                )
                image = F.pad(image, pad_dims, "constant", 0)

                # add correction factor
                from_ += patch_size//2
                to += patch_size//2

            # and finally extract
            patches.append(image[:, :, from_[0]:to[0], from_[1]:to[1]])

        # concatenate into a single tensor
        patches = torch.cat(patches, dim=0)
        return patches

    def denormalize(self, coords, image_size):
        """
        Convert coordinates in the range [-1, 1]/[-1, 1] to
        coordinates in the range [0, H]/[0, W] where `image_size` is
        the size of the image.
        """
        new_cords = (coords + 1.0) / 2
        new_cords *= image_size.float()
        return new_cords.floor().long()

    def exceeds(self, from_, to, image_size):
        """
        Check whether the extracted patch will exceed
        the boundaries of the image of size `image_size`.
        """
        return ((from_[0] < 0) or
                (from_[1] < 0) or
                (to[0] >= image_size[0]) or
                (to[1] >= image_size[1]))