import torch
import torch.jit as jit
from torch import nn
from typing import List


# * ------ Time-varying recurrent neural network class ----------
#   _________      _______  _   _ _   _
#  |__   __\ \    / /  __ \| \ | | \ | |
#     | |   \ \  / /| |__) |  \| |  \| |
#     | |    \ \/ / |  _  /| . ` | . ` |
#     | |     \  /  | | \ \| |\  | |\  |
#     |_|      \/   |_|  \_\_| \_|_| \_|
# * ---------------------------------------------------------------
class RNNRX(jit.ScriptModule):
    def __init__(
        self,
        dev,
        szNNvecpM: torch.Tensor,
        T_rnn: int,
        L_SIC: int,
        S_SIC: int,
        cur_SIC: int,
        f_cplx_mod: bool,
        f_cplx_AWGN: bool,
        f_RNN_bi=1,  # Default bidirectional
    ):
        super().__init__()
        self.dev = dev  # Save device

        ## Check if bidirectional NN is requested
        if f_RNN_bi == True:
            self.mult_BIRNN = 2  # Scale layers with factor of two because of output concatenation of bidirectional RNN
        elif f_RNN_bi == False:
            self.mult_BIRNN = 1  # No need to scale layers

        ## Check if modulation format for SIC is complex-valued
        self.f_cplx_mod = f_cplx_mod  # Scale input layer with factor of x2 because of real composite representation
        self.f_cplx_AWGN = f_cplx_AWGN  # Scale input layer size when noise (and channel outputs) are complex

        self.L_RNN = int(len(szNNvecpM))  # Number of *RNN* layers
        self.L_last_dim = int(szNNvecpM[-2])  # Input dim of last linear layer
        self.L_SIC = L_SIC
        self.S_SIC = S_SIC

        if cur_SIC > 1:  # Adjust input layer sizes depending on SIC stage
            in_dim = (self.f_cplx_AWGN + 1) * szNNvecpM[0] + self.L_SIC * (
                1 + self.f_cplx_mod
            )
        else:
            in_dim = (self.f_cplx_AWGN + 1) * szNNvecpM[0]

        self.models = nn.ModuleList()  # Variable layer list
        for n_dim in szNNvecpM[1:-1:]:
            self.models.append(
                TVRNN(
                    dev,
                    in_dim,
                    n_dim,
                    T_rnn,
                    S_SIC,
                    cur_SIC,
                )
            )

            # Mult x 2 for next input dimension, because bidirectional RNN concatenates two previous output vectors
            in_dim = int(self.mult_BIRNN * n_dim)
        self.Lin_layer = nn.Linear(
            in_dim, int(szNNvecpM[-1])
        )  # Append last linear layer

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for model in self.models:
            x = model(x)

        out = x.contiguous().view(-1, self.mult_BIRNN * self.L_last_dim)
        out = self.Lin_layer(out)

        return out


# * ------------------ Time-Varying RNN Layer ---------------------
class TVRNN(jit.ScriptModule):
    def __init__(
        self,
        dev: str,
        input_size: int,
        hidden_size: int,
        Trnn: int,
        S_SIC: int,
        cur_SIC: int,
    ):
        super().__init__()

        self.inp_sz = int(input_size)
        self.hid_sz = int(hidden_size)
        self.Trnn = int(Trnn)
        self.dev = dev

        if cur_SIC == 1:
            self.Ntv = 1  # Number of time-varying hidden weights
            self.cells_fw = nn.ModuleList(
                [CRNNCell(self.inp_sz, self.hid_sz, self.dev)]
            )
            self.cells_bw = nn.ModuleList(
                [CRNNCell(self.inp_sz, self.hid_sz, self.dev)]
            )

        elif cur_SIC > 1:
            self.Ntv = int(S_SIC - cur_SIC + 1)  # Number of time-varying hidden weights

            # Simplify this
            self.cells_fw = nn.ModuleList(
                [
                    CRNNCell(self.inp_sz, self.hid_sz, self.dev)
                    for count in range(self.Ntv)
                ]
            )
            self.cells_bw = nn.ModuleList(
                [
                    CRNNCell(self.inp_sz, self.hid_sz, self.dev)
                    for count in range(self.Ntv)
                ]
            )

    @jit.script_method
    def recurFW(self, x: torch.Tensor) -> torch.Tensor:
        # Input dim(x) = nBatch x N_Trnn x Nin
        nBatch = x.size()[0]

        # Initialize with uniform random numbers
        # [https://pytorch.org/docs/stable/generated/torch.nn.RNN.html]
        ksc = torch.sqrt(torch.tensor(1 / self.hid_sz))
        h = 2 * ksc * torch.rand(nBatch, self.hid_sz, device=self.dev) - ksc

        inputs = x.unbind(1)  # Unbind Trnn
        outputs = torch.jit.annotate(List[torch.Tensor], [])

        # Iterate over RNN time steps
        for i in range(len(inputs) // self.Ntv):
            for k, tvrnn_cell in enumerate(self.cells_fw):
                # dim(h) = (batch_size, hidden_size)
                h = tvrnn_cell(inputs[i * self.Ntv + k], h)
                outputs += [h]

        # Because formerly, we unbound dim=1 (Trnn)
        return torch.stack(outputs, dim=1)

    @jit.script_method
    def recurBW(self, x: torch.Tensor) -> torch.Tensor:
        # Input dim(x) = nBatch x N_Trnn x Nin
        nBatch = x.size()[0]

        # Initialize with uniform random numbers
        # [https://pytorch.org/docs/stable/generated/torch.nn.RNN.html]
        ksc = torch.sqrt(torch.tensor(1 / self.hid_sz))
        h = 2 * ksc * torch.rand(nBatch, self.hid_sz, device=self.dev) - ksc

        inputs = reverse(x.unbind(1))  # Unbind Trnn
        outputs = torch.jit.annotate(List[torch.Tensor], [])

        # Iterate over RNN time steps
        for i in range(len(inputs) // self.Ntv):
            for k, tvrnn_cell in enumerate(self.cells_bw):
                # dim(h) = (batch_size, hidden_size)
                h = tvrnn_cell(inputs[i * self.Ntv + k], h)
                outputs += [h]

        # Because formerly, we unbound dim=1 (Trnn)
        return torch.stack(reverse(outputs), dim=1)

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Parallelize FW and BW path, as they are independent
        # [https://pytorch.org/tutorials/advanced/torch-script-parallelism.html]

        ## --- Forward path
        future_f = torch.jit.fork(self.recurFW, x)

        ## --- Backward path
        out_bw = self.recurBW(x)

        out_fw = torch.jit.wait(future_f)  # Wait for FW path to finish

        ## -- Return concatenated
        return torch.cat((out_fw, out_bw), dim=2)


# * ----------------------- Custom RNN Cell ---------------------
# References for JIT; Code inspired by:
# [https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns/custom_lstms.py]
# [https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/]
class CRNNCell(jit.ScriptModule):
    def __init__(self, inp_sz: int, hid_sz: int, dev):
        super().__init__()
        self.input_size = inp_sz
        self.hidden_size = hid_sz
        k = torch.sqrt(torch.tensor(1 / hid_sz))  # For uniform initialization
        self.weight_ih = nn.Parameter(
            2 * k * torch.rand(hid_sz, inp_sz, device=dev) - k
        )
        self.weight_hh = nn.Parameter(
            2 * k * torch.rand(hid_sz, hid_sz, device=dev) - k
        )
        self.bias_ih = nn.Parameter(2 * k * torch.rand(hid_sz, device=dev) - k)
        self.bias_hh = nn.Parameter(2 * k * torch.rand(hid_sz, device=dev) - k)

    @jit.script_method
    def forward(self, input: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        hx = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )

        hx = torch.relu(hx)
        return hx


def reverse(lst: List[torch.Tensor]) -> List[torch.Tensor]:
    return lst[::-1]
