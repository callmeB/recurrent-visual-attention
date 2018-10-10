import torch
import torch.nn as nn
import torch.nn.functional as F


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


class glimpse_network(nn.Module):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the rectangular patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of channels in each image.
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, h_g, g, k, s):
        super(glimpse_network, self).__init__()
        self.retina = retina(g, k, s)

        # convolutional layers
        self.glimpse_conv = nn.Sequential(
            nn.Conv2d(3*k, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # glimpse fc layer
        self.glimpse_fc = nn.Sequential(
            nn.Linear(g[0]*g[1]*128, 1024),
            nn.ReLU()
        )

        # location layer
        self.glimpse_loc = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        phi = self.glimpse_conv(phi)
        phi = phi.view(phi.size(0), -1)

        what = self.glimpse_fc(phi)
        where = self.glimpse_loc(l_t_prev)

        # feed to fc layer
        g_t = torch.mul(what, where)
        return g_t


class cnn_network(nn.Module):
    def __init__(self, h_g, h_l, g, k, s, c):
        super(cnn_network, self).__init__()
        self.a = glimpse_network(h_g, g, k, s)
        self.b = glimpse_network(h_g, g, k, s)

    def forward(self, x_a, x_b, l_t_a_prev, l_t_b_prev):
        g_t_a = self.a(x_a, l_t_a_prev)
        g_t_b = self.b(x_b, l_t_b_prev)
        g_t = torch.cat((g_t_a, g_t_b), 1)
        return g_t

class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
    """
    def __init__(self, input_size, hidden_size):
        super(core_network, self).__init__()

        self.rnn_features = nn.GRUCell(input_size, hidden_size)
        self.rnn_location = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h_1 = self.rnn_features(g_t, h_t_prev[0])
        h_2 = self.rnn_location(h_1, h_t_prev[1])
        h_t = (h_1, h_2)
        return h_t


class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - a_t: output probability vector over the classes.
    """
    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - mu: a 2D vector of shape (B, 2).
    - l_t: a 2D vector of shape (B, 2).
    """
    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std


        self.locator_a = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        self.locator_b1 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        self.locator_b2 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, h_t):
        # compute means
        mu_a = torch.tanh(self.locator_a(h_t.detach()))
        mu_b1 = torch.tanh(self.locator_b1(h_t.detach()))
        mu_b2 = torch.tanh(self.locator_b2(h_t.detach()))

        # reparametrization trick
        noise = torch.empty_like(mu_a)
        l_t_a = mu_a + noise.normal_(std=self.std)
        l_t_b1 = mu_b1 + noise.normal_(std=self.std)
        l_t_b2 = mu_b2 + noise.normal_(std=self.std)

        # bound between [-1, 1]
        l_t_a = torch.tanh(l_t_a)
        l_t_b1 = torch.tanh(l_t_b1)
        l_t_b2 = torch.tanh(l_t_b2)

        return mu_a, l_t_a, mu_b1, l_t_b1, mu_b2, l_t_b2


class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        h_t = h_t[0] + h_t[1]
        b_t = F.relu(self.fc(h_t.detach()))
        return b_t
