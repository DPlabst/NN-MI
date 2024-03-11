import time
import commpy as comm
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import asciichartpy
from tabulate import tabulate

from nnmi import rnn  # Time-varying RNN class


# * ----- Raised-cosine TX DAC  -----
def gen_ps_rcos(N_sim, N_span, alph_roll):
    # The modulo makes sure we always have the center tap
    t_even, h_even = comm.rcosfilter(
        N_sim * N_span + (N_sim % 2) + 2, alph_roll, 1, N_sim
    )
    t = t_even[1:]  # Cut away first sample of CIR
    h = h_even[1:]  # Cut away first sample of time vector

    return h, t


# * ----- Root-Raised-cosine TX DAC  -----
def gen_ps_rrcos(N_sim, N_span, alph_roll):
    # The modulo makes sure we always have the center tap
    t_even, h_even = comm.rrcosfilter(
        N_sim * N_span + (N_sim % 2) + 2, alph_roll, 1, N_sim
    )
    t = t_even[1:]  # Cut away first sample of CIR
    h = h_even[1:]  # Cut away first sample of time vector

    return h, t


# * ----- Normalize channel impulse response (CIR) energy to unit power  -----
def norm_ps(N_sim, filt):
    # Normalize energy: sum_i |filt_i|^2 = N_sim
    filt = filt * np.sqrt(N_sim / sum(abs(filt) ** 2))

    if abs(sum(abs(filt) ** 2) - N_sim) > 1e-10:
        raise ValueError("Violating power constraint.")

    return filt


def c2r_horz_ind(a, N):  # Composite-real representation
    a2rind = torch.hstack((a, a + N))
    return a2rind


def norm_unit_var(a, mu_a, P):  # Normalize to unit variance
    return (a - mu_a) / np.sqrt(P)


# * ----- Standard single-mode fiber (SSMF) channel ----
class SSMF:
    def __init__(self, band, L_SSMF, R_sym):
        self.L_SSMF = L_SSMF  # Length of fiber [m]
        self.R_sym = R_sym  # Symbol rate GBd or Gsymb/s
        self.c_light = 299792458  # Speed of light [m/s]

        if band == "C":
            self.band = band  # "C-band"
            self.D = D = 17 * 1e-12 / (1e-9 * 1e3)  # Dispersion coefficient [s/(m*m)]
            self.lambd_c = 1550e-9  # Carrier wavelength [m]

            # Chromatic dispersion [s^2/m] C-Band
            self.beta2 = -self.D * self.lambd_c**2 / (2 * np.pi * self.c_light)

            # Third-order dispersion [s^3/m] #C-Band
            self.beta3 = 0.12 * (1e-12) ** 3 / 1e3

        elif band == "O":
            self.band = band  # "O-band"
            self.lambd_c = 1300e-9  # Carrier wavelength [m]

            # Chromatic dispersion [s^2/m] O-Band
            self.beta2 = -2 * (1e-12) ** 2 / 1e3

            # Third-order dispersion [s^3/m] #O-Band
            self.beta3 = 0.07 * (1e-12) ** 3 / 1e3

    # * ----- Generate combined impulse response (CIR) between TX-DAC (g_ps) and fiber response -----
    def gen_comb_cir(self, N_sim, g_ps):
        if self.L_SSMF > 0:
            N_CD = len(g_ps)  # Length in taps
            deltaf = (N_sim * self.R_sym) / N_CD

            # Frequency spacing vector
            f = (np.arange(N_CD) - np.floor(N_CD / 2)) * deltaf

            # Frequency response of chromatic dispersion and third-order dispersion
            H_omega = np.exp(
                +1j
                * (
                    (2 * np.pi * f) ** 2 * self.beta2 / 2
                    + (2 * np.pi * f) ** 3 * self.beta3 / 6
                )
                * self.L_SSMF
            )

            g_comb = np.fft.ifft(np.fft.ifftshift(H_omega) * np.fft.fft(g_ps))

        elif self.L_SSMF == 0:  # Return g_ps if back-to-back operation is requested
            g_comb = g_ps

        return g_comb


# * ----- Standard single-mode fiber (SSMF) channel ----
class CH6:
    def __init__(self):
        self.L_SSMF = 0  # Dummy
        self.R_sym = 0  # Dummy

        self.h = np.array(
            [0.19, 0.35, 0.46, 0.5, 0.46, 0.35, 0.19]
        )  # Sampled at symbol rate

    # * ----- Generate combined impulse response (CIR) between TX-DAC (g_ps) and fiber response -----
    def gen_comb_cir(self, N_sim, g_ps):

        g_CH6_up = np.zeros(N_sim * (len(self.h) - 1) + 1)
        g_CH6_up[0::N_sim] = self.h
        g_comb = np.convolve(g_ps, g_CH6_up, mode="same")

        return g_comb


# *---------------- Apply channel ------------
# - Symbol-wise precoding
# - Apply filter g[u] to Nsim-fold upsampled (precoded) symbols
# - Apply nonlinearity nonl_f(.)
# - Add AWGN
# - Apply filter h[u] and downsample by factor "d"
class channel:
    def __init__(
        self,
        g_pre,
        g_ps,
        tx_rolloff,
        h,
        rx_cutoff,
        nonl_f,
        phychan,  # Fiber channel instance
        N_sim,
        d,
        f_cplx_AWGN,
        modf_str,
        f_diff_precoding,
    ):
        # * ----- TX Parameters -----
        self.modf_str = modf_str

        # f_cplx_mod: If 0: real modulation, if 1:complex modulation
        self.Xalph, self.M, self.f_cplx_mod = self.gen_mod_alph(modf_str)

        self.g_pre = g_pre
        self.f_diff_precoding = f_diff_precoding

        # Compute "combination" of  SSMF channel and TX DAC g_ps
        self.g = phychan.gen_comb_cir(N_sim, g_ps)
        self.tx_rolloff = tx_rolloff

        # * ----- RX Parameters -----
        self.f_cplx_AWGN = f_cplx_AWGN
        self.h = h
        self.rx_cutoff = rx_cutoff

        # * ----- Nonlinearity -----
        self.nonl_f = nonl_f

        # * ----- Fiber Channel parameter -----
        self.R_sym = phychan.R_sym
        self.L_SSMF = phychan.L_SSMF

        self.N_sim = N_sim
        self.d = d

    # Some standard modulation formats (Note: Only PAM/ASK/SQAM are supported with f_diff_precoding = True)
    def gen_mod_alph(self, modf_str):
        modf_arr = modf_str.split("-")
        M = np.int64(modf_arr[0])

        if modf_arr[1] == "PAM":
            Xalph = np.linspace(0, 1, num=M, endpoint=True)  # PAM
            f_cplx_mod = 0  # Real

        elif modf_arr[1] == "ASK":
            Xalph = np.linspace(-1, 1, num=M, endpoint=True)  # ASK
            f_cplx_mod = 0  # Real

        elif modf_arr[1] == "SQAM":
            f_cplx_mod = 1  # Complex
            X_base = np.array([1, 1j, -1, -1j])  # Star
            Xalph = np.array([], dtype=np.complex64)
            for ii in range(M // len(X_base)):
                Xalph = np.append(Xalph, (ii + 1) * X_base)  # Scale and append

        elif modf_arr[1] == "QAM":  # Must be square
            f_cplx_mod = 1  # Complex
            Mp = int(np.sqrt(M))
            Xalph1D = np.linspace(-1, 1, num=Mp, endpoint=True)  # Mp-ASK
            Xalph = np.reshape(np.add.outer(Xalph1D.T, 1j * Xalph1D), -1)  # M-QAM

        # Normalize
        Xalph = Xalph / np.sqrt(np.mean(np.abs(Xalph) ** 2))

        if abs(np.mean(abs(Xalph) ** 2) - 1) > 1e-10:
            raise ValueError("Violating constellation power constraint.")

        M = len(Xalph)  # Cardinality of output alphabet

        return Xalph, M, f_cplx_mod

    # * ----- Generate input symbols from the chosen constellation -----
    def gen_input(self, n, Ptx):
        M = len(self.Xalph)
        ui = np.random.choice(np.arange(M), n)  # Generate random indices
        u = np.sqrt(Ptx) * self.Xalph[ui]  # Symbols
        u = u.reshape(
            -1,
        )

        # For nonl_f(.) = SLD, differential encoding helps
        # Supported modulation formats: ASK/SQAM
        if self.f_diff_precoding:
            if any(np.angle(self.Xalph)):  # Apply only for phase modulation
                x = self.d_enc(u)  # Differential phase coding

                # Remove spurious imaginary parts to improves numerical stability
                Xalph_sc = np.sqrt(Ptx) * self.Xalph
                outer_sub_max_i = np.argmin(
                    np.abs(np.subtract.outer(x, Xalph_sc)), axis=1
                )
                x2 = Xalph_sc[outer_sub_max_i]

            else:
                x2 = u  # Do nothing is modulation is real
        else:
            x2 = u  # Do nothing

        return ui, u, x2

    # * ---- IIR differential phase encoding ----
    def d_enc(self, u):
        # x_n = u_n*exp(1j*x_(n-1)) # Only supported for ASK and SQAM
        # See also [Appendix, https://arxiv.org/pdf/2212.07761v1.pdf]

        del_shift = 0
        u_ph = np.angle(u)

        # Differentially encode the phase
        enc_phase, _ = scipy.signal.lfilter([1], [1, -1], u_ph, zi=[0])

        # Stitch together with amplitude
        x = np.abs(u) * np.exp(1j * enc_phase + 1j * del_shift)

        return x

    # * ----- Simulate TX, channel and RX --------
    def simulate(self, n, Ptx):
        # Generate input symbols
        ui, u, x = self.gen_input(n, Ptx)

        ## ---- Symbol-wise precoding ----
        xpre = scipy.signal.oaconvolve(x, self.g_pre, "same")

        # Dummy symbol power control
        # if abs(np.mean(abs(xpre)**2)-Ptx) > 5E-2*Ptx: #Allow maximum 5% tolerance in TX power
        #    raise ValueError('Violating constellation power constraint.')

        ## ---- TX DAC ----
        x_up = np.zeros(self.N_sim * n, dtype=np.complex64)
        x_up[:: self.N_sim] = xpre

        # Remark for synchronization: Filters g,h should be symmetric and have odd length
        # Example: h = [h-2,h-1,h0,h1,h2]
        # For N_sim = 1 we obtain:
        # ----------------------------------------------
        # h0  h1  h2 0  0           x0               y0
        # h-1 h0  h1 h2 0     x     x1      :=       y1
        # h-2 h-1 h0 h1 h2          x2               y2
        #  .   .   .  .  .           .               .
        # ----------------------------------------------
        z = scipy.signal.oaconvolve(x_up, self.g, mode="same")  # Apply filter g[u]
        z_f = self.nonl_f(z, Ptx)  # Apply nonlinearity

        # ---- AWGN ----
        # Definition: AWGN PSD Amplitude N0 := 1/B (Double-sided, per dimension)
        # Symbol rate: B
        # -> N_sim = 1: var_n = N0/2* N_sim * B = 1/2 (per dimension)
        # -> N_sim = 2: var_n = N0/2* N_sim * B = 1   (per dimension)
        # -> N_sim = 3: var_n = N0/2* N_sim * B = 3/2 (per dimension)
        # -> N_sim = 4: var_n = N0/2* N_sim * B = 4/2 (per dimension)

        # Definition of N0/2 per dimension
        # B cancels out and can be set to 1 (see table above)
        N0over2 = 1 / 2
        var_n = N0over2 * self.N_sim  # Noise variance per dimension

        if self.f_cplx_AWGN == 0:  # Create real AWGN with variance var_n
            noise = np.sqrt(var_n) * np.random.randn(
                self.N_sim * n,
            )
        elif self.f_cplx_AWGN == 1:  # Create complex AWGN with variance var_n
            noiseR = np.random.randn(
                self.N_sim * n,
            )
            noiseI = np.random.randn(
                self.N_sim * n,
            )

            noise = np.sqrt(var_n) * (noiseR + 1j * noiseI)

        y = z_f + noise  # Add noise

        # Apply RX ADC filter h[u]
        y_Nsim = scipy.signal.oaconvolve(y, self.h, mode="same")

        # Downsample by factor "d" to N_os-fold oversampling
        y_Nos = y_Nsim[:: self.d]

        return ui, u, x, y_Nos


# Training and Validation
# of a SIC stage receiver
# * ---------------------
#    _____ _____ _____
#   / ____|_   _/ ____|
#  | (___   | || |
#   \___ \  | || |
#   ____) |_| || |____
#  |_____/|_____\_____|
# * ---------------------
class SICstage:
    def __init__(
        self,
        vstr,
        cur_SIC,
        mychan,
        dev,
        Ptx_dB_vec,
        szNNvec,
        lr,
        Ni,
        n,
        n_frames,
        n_frames_sched_ver,
        n_batch,
        S_SIC,
        L_SIC,
        L_snr,
        N_os,
        N_sched,
        T_rnn_raw,
        S_SIC_vec_1idx,
    ):
        self.chan = mychan  # Store channel

        self.vstr = vstr  # Version
        self.Ptx_dB_vec = Ptx_dB_vec
        self.Ni = Ni
        self.lr = lr
        self.n_frames = n_frames
        self.n_frames_sched_ver = n_frames_sched_ver
        self.n_batch = n_batch
        self.dev = dev
        self.cur_SIC = cur_SIC
        self.N_os = N_os
        self.S_SIC = S_SIC
        self.L_SIC = L_SIC
        self.L_snr = L_snr
        self.S_SIC_vec_1idx = S_SIC_vec_1idx

        self.N_sched = N_sched

        # * ------- Find next greater integer T_rnn to provided T_rnn_raw ---------
        # Implementation-specific for block-processing in time-varying RNNs
        # Period (order) of time-varying RNN
        self.Ntv = self.S_SIC - cur_SIC + 1

        # Make time-depth integer divisable by time-varying RNN order
        self.T_rnn = (-(-T_rnn_raw // self.Ntv)) * self.Ntv

        # * ------ Create training and validation sequences with integer multiples of first dimensions -----
        n_sc = S_SIC * self.T_rnn  # Blocklength n needs to be integer divisible by n_sc
        n_train = n_batch * self.T_rnn  # Number of TX symbols for training

        # Keep training data constant per SIC stage.
        # Example: When training with blocklength n and S_SIC = 3,
        # cur_SIC = 2 uses only 2/3 of n for training. We rescale for this loss.
        sc_y_eff = 1 / (self.Ntv / self.S_SIC)

        # Make sure we have integer fractions after creating RNN input tensors
        self.n_train_p_stage = int((-(-sc_y_eff * n_train) // n_sc) * n_sc)
        self.n_verif_p_stage = int((-(-sc_y_eff * n) // n_sc) * n_sc)

        # Compute effective layer size due to concatenating at the end of each RNN section.
        self.nVec_eff = np.copy(szNNvec)

        # Double first layer if f_cplx_AWGN = True
        self.nVec_eff[0] = (self.chan.f_cplx_AWGN + 1) * szNNvec[0]

        # Double every other input size due to bidirectional RNN concatenation
        for l in range(1, len(szNNvec)):
            self.nVec_eff[l] = 2 * szNNvec[l]

        # Append output dimension
        self.szNNvecpM = np.append(szNNvec, self.chan.M)

        # Approximate RNN memory
        self.NtildeRNN = int(np.floor(self.szNNvecpM[0] / self.N_os) + self.T_rnn - 1)

        # * --- Index Matrices generate index matrices once and use them to select relevant NN inputs
        # Chunk index matrix of y for TRAINING
        idx_y_ds_train = self.get_y_idx_mat(N_os * self.n_train_p_stage)

        # Chunk index matrix of y for VALIDATION
        idx_y_ds_vali = self.get_y_idx_mat(N_os * self.n_verif_p_stage)

        if cur_SIC > 1:
            # Chunk index matrix of x for TRAINING
            idx_x_sic_train = self.get_sic_idx_mat(self.n_train_p_stage)

            # Chunk index matrix of x for VALIDATION
            idx_x_sic_vali = self.get_sic_idx_mat(self.n_verif_p_stage)

        else:
            idx_x_sic_train = []
            idx_x_sic_vali = []

        # Combined chunk index matrix of [y,x] for TRAINING
        self.idx_mat_train = self.get_comb_idx_mat(
            idx_y_ds_train,
            idx_x_sic_train,
            self.n_train_p_stage,
        )

        # Combined chunk index matrix of [y,x] for VALIDATION
        self.idx_mat_vali = self.get_comb_idx_mat(
            idx_y_ds_vali,
            idx_x_sic_vali,
            self.n_verif_p_stage,
        )

    # * --------- Generate channel outputs y index matrix  --------
    # Note: We store index matrices once and use them to select relevant NN inputs

    def get_y_idx_mat(self, ny):
        Hin = self.szNNvecpM[0]  # Equivalent _complex_ inputs (without SIC)

        idx_y = torch.arange(ny).reshape(-1, 1)
        idx_y_mat = torch.zeros(ny, Hin, dtype=torch.int64)

        # Extend cyclically (Comment: Integer division is OK)
        for j in range(Hin):
            idx_y_mat[:, j : j + 1] = torch.roll(idx_y, j - (Hin) // 2)

        idx_y_mat = torch.fliplr(idx_y_mat)  # Flip for correct time ordering

        # Downsample first dimension by N_os: Take every N_os-th row
        # observation indices corresponding to all TX symbols
        # Dimensions: [ny/Nos x Hin] = [n x Hin]
        chunks_y_ds_idx = idx_y_mat[:: self.N_os]

        # Reshape according to SIC stages:
        # Dimensions of tensor: [n/(S_SIC) x N_SIC_stages x Hin]
        # Special case training: n/(S_SIC) = n_batch
        # Requirement for reshape: n/S_SIC is integer
        chunks_y_ds_idx_SIC = torch.reshape(chunks_y_ds_idx, (-1, self.S_SIC, Hin))

        return chunks_y_ds_idx_SIC

    # * --------- Generate SIC symbols x index matrix  --------
    # Note: We store index matrices once and use them to select relevant NN inputs
    def get_sic_idx_mat(self, n):
        n_vec = np.arange(n)

        cur_SIC = self.cur_SIC
        S_SIC = self.S_SIC
        L_SIC = self.L_SIC

        # Admissible index matrix for SIC
        idx_admiss_mat = np.zeros((self.cur_SIC - 1, n // self.S_SIC), dtype=np.int64)

        # Admissible symbol indices via subsampling of n_vec
        # Find out which indices we may take from previous stages [1,...,s-1]
        for ii in range(cur_SIC - 1):
            idx_admiss_mat[ii, :] = n_vec[ii::S_SIC]

        # Vectorize admissible matrix and sort in ascending order
        idx_admiss_vec_asc = (
            np.sort(idx_admiss_mat.reshape(-1, 1)).squeeze().astype(np.int64)
        )

        # Matrix to keep indices for circular minimum distance to idx = [0,1,.S_SIC-1]
        idx_dcirc_min_mat = np.zeros((S_SIC, L_SIC), dtype=np.int64)

        # Measure circular distance from idx = [0,1,...,S_SIC-1] to admissible vector 'idx_admiss_vec_asc'
        # Example: S=3, s=2
        # | x0, x1, x2 | x3, x4, x5 | x6, x7, x8 | from which we know [..., x0, x3, x6, ...] for the first stage
        # The function findes the "closest" indices I of previously decoded symbols to [x1,x2].
        # Accordingly, the "closest" indices from symbols [x4,x5] are simply I+S_SIC.
        for idx_int in range(0, S_SIC):  # Related index k=0,1,...,S_SIC-1
            i = (idx_int - idx_admiss_vec_asc) % n  # Make circulant
            j = (idx_admiss_vec_asc - idx_int) % n  # Make circulant

            # Sort list: Get "closest" L_SIC symbol indices to index k=0,1,...,S_SIC-1.
            # Comment: Stable is require for deterministic performance for equal element sorting
            argsort_d_circ = np.argsort(np.min([i, j], axis=0), kind="stable")

            # Take only L_SIC elements for IC
            idx_dcirc_min_mat[idx_int, :] = idx_admiss_vec_asc[argsort_d_circ[0:L_SIC]]

        # Goal: Create a batch of SIC matrices -> repeat 'idx_dcirc_min_mat' in (new) first dimension
        # with dim = (Z x N_SIC_stages x L_SIC)
        # - for training: Z = n_batch,
        rep_sic = np.tile(idx_dcirc_min_mat, (n // S_SIC, 1, 1))

        # Create index offset matrix (Z x S_SIC)
        # which is later added to shift rep_sic
        # [0,...Z] * S_SIC*[1,...,1].T (outer product)
        idx_offset = np.multiply.outer(
            np.arange(n // S_SIC),
            S_SIC * np.ones(S_SIC, dtype=np.int64),
        )

        # Repeat in new dimension to match 'rep_sic'
        idx_rep_offset = np.repeat(idx_offset[:, :, np.newaxis], L_SIC, axis=2)

        # Extend circularily at the edges
        idx_sic_full = (rep_sic + idx_rep_offset) % n

        # Keep only un-KNOWN indices
        # Downsample and remove parts of 'idx_sic_full'
        # that are used for SIC with already known data
        # Example: If current stage s=3 / S=4 -> Remove stages s=1-2
        idx_sic_red = idx_sic_full[:, cur_SIC - 1 : S_SIC, :]

        # Just stacks matrices of remaining stages on top
        # This is expected, as observations chunks are also ordered like this
        idx_sic_red_rsh = np.reshape(idx_sic_red, (-1, L_SIC))

        return idx_sic_red_rsh

    # * --------- Combine observation index matrix and SIC index matrix --------
    # Note: We store index matrices once and use them to select relevant NN inputs
    def get_comb_idx_mat(
        self,
        idx_y_ds,  # Index matrix for observations
        idx_x_sic,  # Index matrix for SIC symbols
        n,  # Blocklength
    ):
        Hin = self.szNNvecpM[0]  # Equivalent _complex_ inputs (without SIC)

        idx_x_sic = torch.as_tensor(idx_x_sic, dtype=torch.int64)  # Cast

        if self.cur_SIC > 1:  # Perform SIC
            # Filter out relevant _complex_ observation indices for current SIC stage
            idx_y_ds_sic = idx_y_ds[:, self.cur_SIC - 1 : self.S_SIC, :]

            # * -------- Channel outputs: Convert y -> [yreal, yimag] to composite real and shift indices -------
            # Note: len(y) := Nos*n
            if self.chan.f_cplx_AWGN == 0:  # Real
                idx_y_ds_sic_rsh_r = idx_y_ds_sic

            elif self.chan.f_cplx_AWGN == 1:  # Complex
                # Convert to composite real
                # Stack in third dimension [yreal, yimag]
                # Shift [yimag] indices by len(y) := Nos*n
                idx_y_ds_sic_rsh_r = torch.cat(
                    (idx_y_ds_sic, idx_y_ds_sic + self.N_os * n), dim=2
                )

            # Tensor dimensions: (Z,Hin); For training: Z = n_batch
            # For complex channel outputs: Take doubling of RNN input dimension into account
            idx_y_ds_sic_rsh = torch.reshape(
                idx_y_ds_sic_rsh_r, (-1, (self.chan.f_cplx_AWGN + 1) * Hin)
            )

            # * -------- SIC: Convert to composite real and shift indices ---------
            # Convert x -> [xreal, ximag] to composite real and shift indices
            # The receiver has yx_composite [yreal, yimag, xreal, ximag],
            # therefore we must shift the indices of x by len([yreal,yimag]) = 2 * self.N_os * n:
            if self.chan.f_cplx_mod == 0:  # Real
                # Shift indices by N_os * n * (f_cplx_AWGN+1)
                idx_x_sic_r = idx_x_sic + self.N_os * n * 2

            elif self.chan.f_cplx_mod == 1:  # Complex
                # Shift indices by (N_os*n)*(f_cplx_AWGN+1)
                idx_x_sic_r = c2r_horz_ind(idx_x_sic, n) + self.N_os * n * 2

            # Concatenate observations and SIC symbols
            idx_mat = torch.hstack((idx_y_ds_sic_rsh, idx_x_sic_r))

            # Reshape into 3D tensor [n/T_rnn, T_rnn, (f_cplx_AWGN + 1) * Hin + (self.chan.f_cplx_mod + 1) * self.L_SIC]
            idx_mat_rsh = np.reshape(
                idx_mat,
                (
                    -1,
                    self.T_rnn,
                    (self.chan.f_cplx_AWGN + 1) * Hin
                    + (self.chan.f_cplx_mod + 1) * self.L_SIC,
                ),
            )

        else:  # SDD
            # Channel outputs: Convert to composite real and shift indices
            if self.chan.f_cplx_AWGN == 0:  # Real
                idx_y_ds_sic_rsh_r = idx_y_ds  # Take all real observations

            elif self.chan.f_cplx_AWGN == 1:  # Complex
                # Convert to composite real
                # Stack in third dimension [yreal, yimag]
                # Shift indices by len(y) = Nos*n
                idx_y_ds_sic_rsh_r = torch.cat(
                    (idx_y_ds, idx_y_ds + self.N_os * n), dim=2
                )

            # Reshape into 3D tensor [n/T_rnn, T_rnn, (1+f_cplx_AWGN ) * Hin]
            idx_mat_rsh = np.reshape(
                idx_y_ds_sic_rsh_r, (-1, self.T_rnn, (self.chan.f_cplx_AWGN + 1) * Hin)
            )

        return idx_mat_rsh

    def simulate(
        self,
    ):
        # Preallocation
        SER_vec = np.zeros((self.L_snr))
        I_qXY_vec = np.zeros((self.L_snr))

        tsum = 0

        # Use incremental training over SNRs
        for SNR_i in range(self.L_snr):
            Ptx = 10 ** (self.Ptx_dB_vec[SNR_i] / 10)  # Linear

            # Reset demapper
            self.demapper = rnn.RNNRX(
                self.dev,
                self.szNNvecpM,
                self.T_rnn,
                self.L_SIC,
                self.S_SIC,
                cur_SIC=self.cur_SIC,
                f_cplx_mod=self.chan.f_cplx_mod,
                f_cplx_AWGN=self.chan.f_cplx_AWGN,
            )
            self.demapper.to(self.dev)  # CUDA

            # Define Loss, Optimizer
            self.optimizer = torch.optim.Adam(self.demapper.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.3,
            )

            self.optimizer.param_groups[0]["lr"] = self.lr  # Reset LR

            IqXY_train_vec = np.zeros(self.Ni)

            # Find normalization factors over n_norm symbols
            n_norm = int(20e3)
            _, _, _, tmp_y = self.chan.simulate(
                n_norm,
                Ptx,
            )

            # Compute statistics
            y_mean = np.mean(tmp_y)
            y_var = np.var(tmp_y)  # NumPy: var(x) := 1/N sum_i (x_i-mu)^2
            x_mean = np.mean(np.sqrt(Ptx) * self.chan.Xalph)
            x_var = np.var(np.sqrt(Ptx) * self.chan.Xalph)

            # Train over Ni batches
            for j in range(self.Ni):
                tstart = time.time()
                idx_u, _, x, y = self.chan.simulate(self.n_train_p_stage, Ptx)

                # Casting and normalization
                x = torch.as_tensor(x)
                x = norm_unit_var(x.type(torch.complex64), x_mean, x_var)
                # SIC chunks in composite real
                x2r = torch.hstack((x.real, x.imag))

                y = torch.as_tensor(y)
                y = norm_unit_var(y.type(torch.complex64), y_mean, y_var)
                # Observation chunks in composite real
                y2r = torch.hstack((y.real, y.imag))
                # Observation chunks and SIC chunks
                yx2r = torch.hstack((y2r, x2r))

                # Take chunks according to indices
                chunks = torch.take(yx2r, self.idx_mat_train)

                # Processes _all_ remaining stages [s, s+1, ... S]
                chunks = chunks.to(self.dev)  # Transfer
                idx_u_hat_soft = self.demapper(chunks)
                # Downsample according to SIC index: take only estimates for current stage [s]
                idx_u_sic_hat_soft = idx_u_hat_soft[
                    0 :: self.S_SIC - (self.cur_SIC - 1)
                ]
                # Take true data for stage [s]
                idx_u_sic = torch.as_tensor(idx_u[self.cur_SIC - 1 :: self.S_SIC])
                idx_u_sic = idx_u_sic.to(self.dev)  # Transfer

                # Compare via CE
                ce = F.cross_entropy(idx_u_sic_hat_soft, idx_u_sic)
                # Training rate: [bits/per channel use]
                IqXY_train_vec[j] = np.log2(self.chan.M) - ce * np.log2(np.exp(1))

                self.optimizer.zero_grad(set_to_none=True)
                ce.backward()
                self.optimizer.step()

                # Learning rate:
                if j % self.N_sched == 0:
                    _, IqXY_i_val = self.get_metrics(
                        Ptx, y_mean, y_var, x_mean, x_var, self.n_frames_sched_ver
                    )
                    self.scheduler.step(IqXY_i_val)
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    if cur_lr < 1e-5:
                        break

                tend = time.time()  # Time needed for j iterations,
                tsum = tsum + (
                    tend - tstart
                )  # Only measure the training, neglect validation

                if j % self.N_sched == 0:  # Print back every ...
                    self.printprogress(
                        I_qXY_vec, tsum, SNR_i, IqXY_train_vec, j, cur_lr
                    )

            # Compute validation metrics
            SER_i, IqXY_i = self.get_metrics(
                Ptx, y_mean, y_var, x_mean, x_var, self.n_frames
            )

            SER_vec[SNR_i] = SER_i
            I_qXY_vec[SNR_i] = IqXY_i

        return SER_vec, I_qXY_vec, self.gen_filename(), self.complexity_rnn()

    # * -------- NN Validation --------
    def get_metrics(self, Ptx, y_mean, y_var, x_mean, x_var, n_frames):
        SER_vec_realz = np.zeros(n_frames)
        IqXY_vec_realz = np.zeros(n_frames)

        for i in range(n_frames):
            # Create new data for evaluation
            # Scale modulation alphabet depending on SNR (b ydefinition, we keep the noise power constant)
            idx_u, _, x, y = self.chan.simulate(self.n_verif_p_stage, Ptx)

            # Casting and normalization of batch (the error metrics only use indices)
            x = torch.as_tensor(x)
            x = norm_unit_var(x.type(torch.complex64), x_mean, x_var)
            # SIC chunks in composite real
            x2r = torch.hstack((x.real, x.imag))

            y = torch.as_tensor(y)
            y = norm_unit_var(y.type(torch.complex64), y_mean, y_var)
            # Observation chunks in composite real
            y2r = torch.hstack((y.real, y.imag))
            # Observation chunks and SIC chunks
            yx2r = torch.hstack((y2r, x2r))

            chunks = torch.take(yx2r, self.idx_mat_vali)

            # Processes _all_ remaining stages [s, s+1, ... S]
            chunks = chunks.to(self.dev)  # Transfer
            idx_u_hat_soft = self.demapper(chunks)

            # Downsample according to SIC index: take only estimates for current stage [s]
            idx_u_sic_hat_soft = idx_u_hat_soft[0 :: self.S_SIC - (self.cur_SIC - 1)]

            # Rate
            idx_u_sic = torch.as_tensor(idx_u[self.cur_SIC - 1 :: self.S_SIC])
            idx_u_sic = idx_u_sic.to(self.dev)  # Transfer to device

            ce = F.cross_entropy(idx_u_sic_hat_soft, idx_u_sic)  # CE = -log(softmax)

            # Rate [bits/per channel use]
            IqXY_tmp = np.log2(len(self.chan.Xalph)) - ce * np.log2(np.exp(1))

            # SER
            idc_u_hat_hard = torch.argmax(idx_u_sic_hat_soft, axis=1)
            SER_tmp = torch.mean((idc_u_hat_hard != idx_u_sic).to(float))

            IqXY_vec_realz[i] = IqXY_tmp
            SER_vec_realz[i] = SER_tmp

        SER = np.mean(SER_vec_realz)
        IqXY = np.mean(IqXY_vec_realz)

        return SER, IqXY

    ## * ---- Print progress in training loop  -----
    def printprogress(
        self,
        I_qXY_vec,
        tsum,
        SNR_i,
        IqXY_train_vec,
        j,
        cur_lr,  # current learning rate
    ):
        # Estimator for remaining simulation time
        delta_t = tsum / ((j + 1) + self.Ni * (SNR_i))
        eta_s = delta_t * (self.Ni * self.L_snr - ((j + 1) + self.Ni * (SNR_i)))
        eta_d = int(eta_s / (24 * 3600))  # ETA days

        # ETA formated in d:X HH:MM:SS
        eta_hms = time.strftime("%H:%M:%S", time.gmtime(eta_s - eta_d * (24 * 3600)))

        # compute real multiplications per APP estimate and SIC stage
        c_mult = self.complexity_rnn()

        # Print back

        # Last 10 SGD steps: 
        plt_range = np.linspace(start=max(0, j - 10), stop=j, num=10).astype(np.int64)
        SNR_list = self.Ptx_dB_vec.tolist()
        SNR_list.insert(0, "SNR")
        Rate_list = I_qXY_vec[0:SNR_i].tolist()
        Rate_list.insert(0, "Rate")
        Rate_list.append("X")
        print("---------------")
        print("Filename: \t" + self.gen_filename())
        print("Stage: \t\t" + str(self.cur_SIC) + "/" + str(self.S_SIC))
        print("Eta stage: \t" + "d:" + str(eta_d) + " " + eta_hms)
        print("Iter.: \t\t" + str(j) + "/" + str(self.Ni))
        print(
            "Eff. train: \t"
            + str(self.n_train_p_stage * (self.S_SIC - self.cur_SIC + 1) / self.S_SIC)
            + " symbols p. stage"
        )
        print("Complexity: \t" + "{:.2E}".format(c_mult) + " mult.")
        print(
            tabulate(
                [SNR_list, Rate_list],
                floatfmt=".3f",
                tablefmt="plain",
                numalign="right",
                stralign="right",
            )
        )
        print("---------------")
        print(
            asciichartpy.plot(
                np.array(
                    [I_qXY_vec, np.ones((self.L_snr)) * np.log2(self.chan.M)]
                ).tolist(),
                {
                    "minimum": 0,
                    "height": 12,
                    "colors": [asciichartpy.blue, asciichartpy.green],
                },
            )
        )
        print(
            "Last SGD: "
            + np.array_str(IqXY_train_vec[plt_range], precision=3, suppress_small=True)
        )
        print("LR: " + str(cur_lr))

    # Generate filename for saving results
    def gen_filename(
        self,
    ):
        fsep = ","  # Field separator

        # Compute NN complexity in #mult per APP estimate (excludes factor S_SIC)
        c_mult = self.complexity_rnn()

        if len(self.S_SIC_vec_1idx) == self.S_SIC:
            indiv_stage_str = ""
        else:
            indiv_stage_str = (
                fsep
                + "s="
                + np.array2string(self.S_SIC_vec_1idx, separator="_")
                .replace("[", "")
                .replace("]", "")
            )

        # Construct filename from settings
        filename = (
            self.vstr  # Version
            + fsep
            + self.chan.modf_str  # Modulation
            + fsep
            + np.array2string(self.nVec_eff, separator="_")
            .replace(" ", "")  # Remove whitespaces
            .replace("[", "")
            .replace("]", "")  # Plot effective layer size vector
            + fsep
            + "lr="
            + "{:.0E}".format(self.lr)  # Learning rate
            + fsep
            + "Nt="
            + str(self.NtildeRNN)  # Approximate RNN memory
            + fsep
            + "Tr="
            + str(self.T_rnn)  # Number of inputs sequential inputs
            + fsep
            + "B="
            + str(int(self.n_batch))  # Batch size
            + fsep
            + "S="
            + str(self.S_SIC)  # Number of SIC stages
            + indiv_stage_str  # Optional: When considering an individual stage
            + fsep
            + "Ls="
            + str(self.L_SIC)  # Number of SIC symbols
            + fsep
            + "I="
            + "{:.1E}".format(self.Ni)  # Number of iterations
            + fsep
            + "V="
            + "{:.1E}".format(self.n_frames)  # Number of frames
            + fsep
            + "n="
            + "{:.1E}".format(self.n_verif_p_stage)  # Number of symbols per frame
            + fsep
            + "a="
            + str(self.chan.tx_rolloff)  # Rolloff factor of DAC filter
            + fsep
            + "Nsim="
            + str(self.chan.N_sim)  # Downsample from Nsim
            + fsep
            + "d="
            + str(self.chan.d)  # Downsample from Nsim
            + fsep
            + "r="
            + str(self.chan.rx_cutoff)  # Relative cut-off (to Nsim) of ADC at RX
            + fsep
            + "L="
            + "{:.1E}".format(self.chan.L_SSMF)  # Length of SSMF [m]
            + fsep
            + "Rs="
            + "{:.1E}".format(self.chan.R_sym)  # Symbol rate [Bd]
            + fsep
            + "C="
            + "{:.1E}".format(c_mult)  # Complexity in real multiplications
            + ".txt"
        )

        return filename

    # * ---- Compute number of _real_ multiplications per SIC stage and per APP estimate -----
    def complexity_rnn(self):
        w = 0  # Init

        # Complexity computation is based on szNNvec with M appended
        szNNvecpM = self.szNNvecpM

        for i in range(1, len(szNNvecpM) - 1):
            if i == 1:
                # Layer 1
                L_im1prime = (1 + self.chan.f_cplx_AWGN) * szNNvecpM[i - 1] + (
                    1 + self.chan.f_cplx_mod
                ) * self.L_SIC
            else:
                L_im1prime = 2 * szNNvecpM[i - 1]  # Layer 2, 3, ....

            L_i = szNNvecpM[i]
            w = w + (L_i * L_im1prime + L_i**2)

        # Last term is the linear output layer
        w = 2 * w + (2 * szNNvecpM[-2]) * szNNvecpM[-1]
        return w
