import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

def GraFIT(r, y, u, W, freqIdentBand, Fs, L, plotResults):
    """
    Disclaimer: The toolbox has been developed and extensively tested in MATLAB and translated to Python. Testing in Python has been limited and some bugs may still be present. Please report any bugs to the author.

    Computes the frequency response of a dynamical system G based on 
    (filtered) white noise perturbations using the Local Rational Model (LRM) 
    method for SISO/MIMO and open-loop/closed-loop systems.

    For open-loop identification, the transfer function between input r and
    output y is computed. For closed-loop identification, the 3-point method
    is used by computing G = Gry/Gru. 

    Args:
        r (numpy.ndarray): 1 x nr x N input matrix (typically external noise signal).
        y (numpy.ndarray): ny x nr x N output matrix.
        u (numpy.ndarray): nu x nr x N output matrix (leave empty for open-loop identification or take plant correction signal for closed-loop identification).
        W (int): local frequency window size (number of neighbouring frequency bins).
        freqIdentBand (tuple or list): 1 x 2 vector defining the frequency band in which to perform the identification.
        Fs (float): Sampling frequency.
        L (tuple or list): 1 x 3 vector with L = [La, Lb, Ld] with
            La: Plant numerator order (default = 2).
            Lb: Transient numerator order (default = 2).
            Ld: Plant and Transient denominator order (default = 2).
        plotResults (bool): [True|False] to visualize results.

    Returns:
        tuple: A tuple containing the following:
            G (numpy.ndarray): Estimate of G with size ny x nr/nu.
            G_var (numpy.ndarray): Variance of G with size ny x nr/nu.
            Gry (numpy.ndarray): Estimate of Gry with size ny x nr (only for CL identification).
            Gru (numpy.ndarray): Estimate of Gru with size nu x nr (only for CL identification).
            Gry_var (numpy.ndarray): Variance of Gry with size ny x nr (only for CL identification).
            Gru_var (numpy.ndarray): Variance of Gru with size nu x nr (only for CL identification).
            Y_contr (dict): Output DFT Y and DFT of input contributions G*R, T and V as struct for G, Gru, Gry with
                G: DFT of Y, G*R, T and V for open-loop case y/r.
                Gry: DFT of Y, G*R, T and V for closed-loop case y/r.
                Gru: DFT of U, G*R, T and V for closed-loop case u/r.

    MIT License - See LICENSE file in the root directory for details.
    Copyright (c) 2025 Mathyn van Dael
    For any questions/bugs, mail m.r.v.dael@tue.nl
    """
    # Detect if SISO inputs and wrong input size (to simplify SISO use), otherwise check if MIMO formatting is correct
    if r.ndim == 1:
        r = np.reshape(r, (1, 1, len(r)))
    elif r.shape[0] > r.shape[2] or r.shape[1] > r.shape[2]:
        raise ValueError('Input matrix r is not the correct size. It should have size 1 x nr x Q with Q the number of datapoints')

    if y.ndim == 1:
        y = np.reshape(y, (1, 1, len(y)))
    elif y.shape[0] > y.shape[2] or y.shape[1] > y.shape[2]:
        raise ValueError('Output matrix y is not the correct size. It should have size ny x nr x Q with Q the number of datapoints')

    if u.size == 0:
        pass
    elif u.ndim == 1:
        u = np.reshape(u, (1, 1, len(u)))
    elif u.shape[0] > u.shape[2] or u.shape[1] > u.shape[2]:
        raise ValueError('Output matrix u is not the correct size. It should have size nu x nr x Q with Q the number of datapoints')
    
    # Initialize variables
    N = r.shape[2]  # Number of datapoints
    freqVec = Fs / N * np.arange(N // 2 + 1)  # Define frequency vector
    nr = r.shape[1]  # Number of inputs (noise signals)
    ny = y.shape[0]  # Number of outputs (error signals)
    nu = u.shape[0]  # Number of outputs (correction signals)

    if L is None or len(L) == 0:  # Check if coefficients are given, if not then set default
        L = [2, 2, 2]  # Default values for [La, Lb, Ld]
    La, Lb, Ld = L

    # Check if frequency window is sufficiently large
    if W <= ((La + 1) + Lb + ny * Ld) // 2:
        W = ((La + 1) + Lb + ny * Ld) // 2 + 1
        print(f'Frequency window has been changed to W = {W}')

    # Get indices of starting and end frequency
    fStartIdx = np.argmin(np.abs(freqVec - freqIdentBand[0]))
    fEndIdx = np.argmin(np.abs(freqVec - freqIdentBand[1]))

    # Check if starting frequency bin + window size is not smaller than lowest frequency bin
    if fStartIdx - W < 0:
        fStartIdx = W + 1

    if fEndIdx + W > (N // 2):
        fEndIdx = N // 2 - W

    # Define identified frequency range vector (add this before the arrays initialization)
    freqVecIdent = freqVec[fStartIdx:fEndIdx]
    nFreq = len(freqVecIdent)

    # Initialize arrays with correct dimensions including frequency points
    G = np.zeros((ny, nr, nFreq), dtype=complex)
    G_var = np.zeros((ny, nr, nFreq), dtype=complex)
    Y_contr = {
        'Y': np.zeros((ny, nr, nFreq), dtype=complex),
        'GR': np.zeros((ny, nr, nFreq), dtype=complex),
        'T': np.zeros((ny, nr, nFreq), dtype=complex),
        'V': np.zeros((ny, nr, nFreq), dtype=complex)
    }

    if u.size == 0:
        for iIn in range(nr):
            G_temp, G_var_temp, Y_contr_temp, G_cov, _ = LRM_SIMO(np.squeeze(r[0, iIn, :]), np.squeeze(y[:, iIn, :]), W, freqVec, [fStartIdx, fEndIdx], La, Lb, Ld, N)
            G[:, iIn, :] = G_temp
            G_var[:, iIn, :] = G_var_temp
            # Update Y_contr values for each output
            for iOut in range(ny):
                Y_contr['Y'][iOut, iIn, :] = Y_contr_temp[iOut]['Y']
                Y_contr['GR'][iOut, iIn, :] = Y_contr_temp[iOut]['GR']
                Y_contr['T'][iOut, iIn, :] = Y_contr_temp[iOut]['T']
                Y_contr['V'][iOut, iIn, :] = Y_contr_temp[iOut]['V']

    else:
        # Initialize matrices
        Grz_cov = np.zeros(((ny**2+nu**2), (ny**2+nu**2), nFreq), dtype=complex)
        G_cov = np.zeros((ny**2, nu**2, nFreq), dtype=complex)
        G_var = np.zeros((ny, nu, nFreq), dtype=complex)

        # Initialize arrays
        Grz = np.zeros((ny * 2, nr, nFreq), dtype=complex)
        Grz_var = np.zeros((ny * 2, nr, nFreq), dtype=complex)
        Grz_Y_contr = [{} for _ in range(ny * 2)]
        
        for iIn in range(nr):
            # Create input vector z = [y; u]
            z = np.vstack([np.squeeze(y[:, iIn, :]), np.squeeze(u[:, iIn, :])])
            
            # Call LRM_SIMO with the combined [y; u] vector
            Grz_temp, Grz_var_temp, Grz_Y_contr_temp, Grz_cov_temp, freqVecIdent = LRM_SIMO(np.squeeze(r[0, iIn, :]), z, W, freqVec, [fStartIdx, fEndIdx], La, Lb, Ld, N)
            
            Grz[:, iIn, :] = Grz_temp
            Grz_var[:, iIn, :] = Grz_var_temp
            
            # Update Grz_Y_contr values
            for i in range(ny * 2):
                Grz_Y_contr[i][iIn] = Grz_Y_contr_temp[i]
            
            # Update covariance matrix
            idx_start = iIn * ny * 2
            idx_end = (iIn + 1) * ny * 2
            Grz_cov[idx_start:idx_end, idx_start:idx_end, :] = Grz_cov_temp
        
        # Extract y and u outputs from Z vector
        Gry = Grz[:ny, :, :]
        Gry_var = Grz_var[:ny, :, :]
        Y_contr['Gry'] = [Grz_Y_contr[i] for i in range(ny)]
        
        Gru = Grz[ny:, :, :]
        Gru_var = Grz_var[ny:, :, :]
        Y_contr['Gru'] = [Grz_Y_contr[i+ny] for i in range(nu)]
        
        # Calculate G = Gry * inv(Gru)
        G = np.zeros((ny, nr, nFreq), dtype=complex)
        for iFreq in range(nFreq):
            G[:, :, iFreq] = np.dot(Gry[:, :, iFreq], np.linalg.inv(Gru[:, :, iFreq]))
        
        # Compute variance of G based on covariance of closed-loop system
        for iFreq in range(nFreq):
            C = np.kron(np.linalg.inv(Gru[:, :, iFreq]).T, np.hstack([np.eye(ny), -G[:, :, iFreq]]))
            G_cov[:, :, iFreq] = np.dot(np.dot(C, Grz_cov[:, :, iFreq]), C.conj().T)
            G_var[:, :, iFreq] = np.reshape(np.diag(G_cov[:, :, iFreq]), (ny, ny))


    if plotResults:

        plt.figure(1)
        fig, axes = plt.subplots(ny, nr, figsize=(15, 10))
        axes = axes.flatten() if ny * nr > 1 else [axes]

        for iOut in range(ny):
            for iIn in range(nr):
                ax = axes[iOut * nr + iIn]

                # Extract G and variance for this input-output pair
                G_Mag = np.abs(G[iOut, iIn, :])

                # Compute confidence bounds
                R = np.sqrt(3) * np.sqrt(G_var[iIn, iOut, :])
                G_Upper = G_Mag + R
                G_Lower = G_Mag - R
                G_Lower[G_Lower < 1e-12] = 1e-12  # Avoid negative lower bounds

                # Plot bounds and transfer function
                confFreq = np.concatenate([freqVecIdent, freqVecIdent[::-1]])
                confRadius = np.concatenate([G_Upper, G_Lower[::-1]])

                ax.plot(freqVecIdent, G_Mag, linewidth=2, color=[0.7, 0.0, 0.0])
                ax.fill(confFreq, confRadius, color=[0.7, 0.0, 0.0], alpha=0.3)
                ax.set_ylim([0.1 * 10**np.floor(np.log10(np.min(G_Mag))), 10 * 10**np.ceil(np.log10(np.max(G_Mag)))])
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True)

        plt.tight_layout()
        plt.show()

    if u.size != 0:
        plt.figure(2)
        fig, axes = plt.subplots(nu, nr, figsize=(15, 10))
        axes = axes.flatten() if nu * nr > 1 else [axes]

        for iOut in range(nu):
            for iIn in range(nr):
                ax = axes[iOut * nr + iIn]

                # Extract G and variance for this input-output pair
                Gru_Mag = np.abs(Gru[iOut, iIn, :])

                # Compute confidence bounds
                R = np.sqrt(3) * np.sqrt(Gru_var[iOut, iIn, :])
                Gru_Upper = Gru_Mag + R
                Gru_Lower = Gru_Mag - R
                Gru_Lower[Gru_Lower < 1e-12] = 1e-12  # Avoid negative lower bounds

                # Plot bounds and transfer function
                confFreq = np.concatenate([freqVecIdent, freqVecIdent[::-1]])
                confRadius = np.concatenate([Gru_Upper, Gru_Lower[::-1]])

                ax.plot(freqVecIdent, Gru_Mag, linewidth=2, color=[0.7, 0.0, 0.0])
                ax.fill(confFreq, confRadius, color=[0.7, 0.0, 0.0], alpha=0.3)
                ax.set_ylim([0.1 * 10**np.floor(np.log10(np.min(Gru_Mag))), 10 * 10**np.ceil(np.log10(np.max(Gru_Mag)))])
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True)

        for iIn in range(nu):
            axes[iIn].set_title(f'Input {iIn + 1}', fontsize=14, fontweight='bold')
            axes[(iIn - 1) * ny + 1].set_ylabel(f'Output {iIn + 1}\nMagnitude [dB]', fontsize=14, fontweight='bold')
            axes[iIn + nu * (ny - 1)].set_xlabel('Frequency [Hz]', fontsize=16, fontweight='bold')

        plt.suptitle('Frequency response magnitude with uncertainty for Gru', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()

        plt.figure(3)
        fig, axes = plt.subplots(ny, nr, figsize=(15, 10))
        axes = axes.flatten() if ny * nr > 1 else [axes]

        for iOut in range(ny):
            for iIn in range(nr):
                ax = axes[iOut * nr + iIn]

                # Extract G and variance for this input-output pair
                Gry_Mag = np.abs(Gry[iOut, iIn, :])

                # Compute confidence bounds
                R = np.sqrt(3) * np.sqrt(Gry_var[iOut, iIn, :])
                Gry_Upper = Gry_Mag + R
                Gry_Lower = Gry_Mag - R
                Gry_Lower[Gry_Lower < 1e-12] = 1e-12  # Avoid negative lower bounds

                # Plot bounds and transfer function
                confFreq = np.concatenate([freqVecIdent, freqVecIdent[::-1]])
                confRadius = np.concatenate([Gry_Upper, Gry_Lower[::-1]])

                ax.plot(freqVecIdent, Gry_Mag, linewidth=2, color=[0.7, 0.0, 0.0])
                ax.fill(confFreq, confRadius, color=[0.7, 0.0, 0.0], alpha=0.3)
                ax.set_ylim([0.1 * 10**np.floor(np.log10(np.min(Gry_Mag))), 10 * 10**np.ceil(np.log10(np.max(Gry_Mag)))])
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True)

        for iIn in range(nu):
            axes[iIn].set_title(f'Input {iIn + 1}', fontsize=14, fontweight='bold')
            axes[(iIn - 1) * ny + 1].set_ylabel(f'Output {iIn + 1}\nMagnitude [dB]', fontsize=14, fontweight='bold')
            axes[iIn + nu * (ny - 1)].set_xlabel('Frequency [Hz]', fontsize=16, fontweight='bold')

        plt.suptitle('Frequency response magnitude with uncertainty for Gry', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()

        plt.figure(4)
        fig, axes = plt.subplots(nu, nr, figsize=(15, 10))
        axes = axes.flatten() if nu * nr > 1 else [axes]

        for iOut in range(nu):
            for iIn in range(nr):
                ax = axes[iOut * nr + iIn]
                ax.scatter(freqVecIdent, np.abs(Y_contr['Gru'][iOut][iIn]['Y']), s=30, color=[0.0, 0.5, 0.0], label='$U$', alpha=0.7)
                ax.scatter(freqVecIdent, np.abs(Y_contr['Gru'][iOut][iIn]['GR']), s=20, color=[0.0, 0.6, 0.6], label='$G_{ru} \cdot R$', alpha=0.7)
                ax.scatter(freqVecIdent, np.abs(Y_contr['Gru'][iOut][iIn]['T']), s=20, color=[0.7, 0.7, 0.0], label='$T$', alpha=0.7)
                ax.scatter(freqVecIdent, np.abs(Y_contr['Gru'][iOut][iIn]['V']), s=20, color=[0.5, 0.0, 0.5], label='$V$', alpha=0.7)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True)
                ax.legend(loc='best', fontsize=10)
                ax.set_title(f'Input {iIn + 1}', fontsize=14, fontweight='bold')
                if iIn == 0:
                    ax.set_ylabel(f'Output {iOut + 1}\nMagnitude [dB]', fontsize=14, fontweight='bold')
                if iOut == nu - 1:
                    ax.set_xlabel('Frequency [Hz]', fontsize=16, fontweight='bold')

        plt.suptitle('Output contributions for Gru', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()

        plt.figure(5)
        fig, axes = plt.subplots(ny, nr, figsize=(15, 10))
        axes = axes.flatten() if ny * nr > 1 else [axes]

        for iOut in range(ny):
            for iIn in range(nr):
                ax = axes[iOut * nr + iIn]
                ax.scatter(freqVecIdent, np.abs(Y_contr['Gry'][iOut][iIn]['Y']), s=30, color=[0.0, 0.5, 0.0], label='$Y$', alpha=0.7)
                ax.scatter(freqVecIdent, np.abs(Y_contr['Gry'][iOut][iIn]['GR']), s=20, color=[0.0, 0.6, 0.6], label='$G_{ry} \cdot R$', alpha=0.7)
                ax.scatter(freqVecIdent, np.abs(Y_contr['Gry'][iOut][iIn]['T']), s=20, color=[0.7, 0.7, 0.0], label='$T$', alpha=0.7)
                ax.scatter(freqVecIdent, np.abs(Y_contr['Gry'][iOut][iIn]['V']), s=20, color=[0.5, 0.0, 0.5], label='$V$', alpha=0.7)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True)
                ax.legend(loc='best', fontsize=10)
                ax.set_title(f'Input {iIn + 1}', fontsize=14, fontweight='bold')
                if iIn == 0:
                    ax.set_ylabel(f'Output {iOut + 1}\nMagnitude [dB]', fontsize=14, fontweight='bold')
                if iOut == ny - 1:
                    ax.set_xlabel('Frequency [Hz]', fontsize=16, fontweight='bold')

        plt.suptitle('Output contributions for Gry', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()

    else:
        plt.figure(6)
        fig, axes = plt.subplots(ny, nr, figsize=(15, 10))
        axes = axes.flatten() if ny * nr > 1 else [axes]
    
        for iOut in range(ny):
            for iIn in range(nr):
                ax = axes[iOut * nr + iIn] 
                ax.scatter(freqVecIdent, np.abs(Y_contr['Y'][iOut][iIn]), s=30, color=[0.0, 0.5, 0.0], label='$Y$', alpha=0.7)
                ax.scatter(freqVecIdent, np.abs(Y_contr['GR'][iOut][iIn]), s=20, color=[0.0, 0.6, 0.6], label='$G_{ry} \cdot R$', alpha=0.7)
                ax.scatter(freqVecIdent, np.abs(Y_contr['T'][iOut][iIn]), s=20, color=[0.7, 0.7, 0.0], label='$T$', alpha=0.7)
                ax.scatter(freqVecIdent, np.abs(Y_contr['V'][iOut][iIn]), s=20, color=[0.5, 0.0, 0.5], label='$V$', alpha=0.7)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True)
                ax.legend(loc='best', fontsize=10)
                ax.set_title(f'Input {iIn + 1}', fontsize=14, fontweight='bold')
                if iIn == 0:
                    ax.set_ylabel(f'Output {iOut + 1}\nMagnitude [dB]', fontsize=14, fontweight='bold')
                if iOut == ny - 1:
                    ax.set_xlabel('Frequency [Hz]', fontsize=16, fontweight='bold')

        plt.suptitle('Output contributions for G', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return G, G_var, Gry, Gru, Gry_var, Gru_var, Y_contr


def LRM_SIMO(r, y, W, freqVec, freqIdx, La, Lb, Ld, N):

    # Ensure r and y are numpy arrays and are row vectors
    r = np.atleast_2d(r)
    y = np.atleast_2d(y)


    if r.shape[0] != 1:
        r = r.T
    if y.shape[0] > y.shape[1]:
        y = y.T

    fStartIdx, fEndIdx = freqIdx[0], freqIdx[1]

    ny = y.shape[0]

    # Create FFT of signals
    R_twoSided = np.fft.fft(r) / np.sqrt(2 * N)
    Y_twoSided = np.fft.fft(y, axis=1) / np.sqrt(2 * N)
    R = R_twoSided[:, :N // 2 + 1]
    Y = Y_twoSided[:, :N // 2 + 1]

    # Define identified frequency range vector
    freqVecIdent = freqVec[fStartIdx:fEndIdx]
    nIdentFreqPoints = len(range(fStartIdx, fEndIdx))

    # Initialize empty matrices for variables
    G = np.zeros((ny, nIdentFreqPoints), dtype=complex)
    T = np.zeros((ny, nIdentFreqPoints), dtype=complex)
    V = np.zeros((ny, nIdentFreqPoints), dtype=complex)
    G_var = np.zeros((ny, nIdentFreqPoints), dtype=complex)
    G_cov = np.zeros((ny, ny, nIdentFreqPoints), dtype=complex)

    # Define vector containing window numbers
    winVec = np.arange(-W, W + 1)

    # Precompute static components of Kn
    Ka = np.power(winVec, np.arange(La + 1)[:, None])                     # Input-related static terms
    Kb = np.power(winVec, np.arange(Lb + 1)[:, None])                     # Transient-related static terms
    Kd = -np.power(winVec, np.repeat(np.arange(1, Ld + 1), ny)[:, None])  # Output-related static terms

    for kIdx in range(nIdentFreqPoints):
        k = kIdx + fStartIdx

        # Define the local frequency window
        localFreqWin = k + winVec

        # Ensure indices are within bounds
        localFreqWin = np.clip(localFreqWin, 0, R.shape[1] - 1)

        # Obtain local Input/Output data
        Rw = R[:, localFreqWin]
        Yw = Y[:, localFreqWin]

        # Create Kw matrix
        Kw = np.vstack([
            Ka * Rw,                   # Input U
            Kb,                        # Transient T
            Kd * np.tile(Yw, (Ld, 1))  # Output Y
        ])

        # Apply scaling (Pintelon 2012, eq. 7-25)
        Dscale = np.diag(np.linalg.norm(Kw, axis=1, ord=2))
        Kw = np.linalg.solve(Dscale, Kw)

        # Assuming Kw and Yw are already defined as NumPy arrays
        U, S, Vt = np.linalg.svd(Kw.conj().T, full_matrices=True)  # SVD of the transpose of Kw
        S = np.diag(S)
        if S.shape[0] < U.shape[1]:
            S = np.pad(S, ((0, U.shape[1] - S.shape[0]), (0, 0)), mode='constant')
        Theta = Yw @ U @ np.linalg.pinv(S.conj().T) @ Vt

        # Compute least squares solution using standard pseudo-inverse
        # Theta = np.dot(Yw, np.linalg.pinv(Kw))

        # Compute residuals before scaling Theta back
        Vn = Yw - Theta @ Kw

        # Rescale Theta and Kw
        Theta = Theta @ np.linalg.inv(Dscale)
        Kw = Dscale @ Kw

        # Store estimated G, T and V
        G[:, kIdx] = Theta[:ny, 0]
        T[:, kIdx] = Theta[:ny, La + 1]  # MATLAB indexing starts at 1
        V[:, kIdx] = Vn[:ny, W]

        # Compute (co-)variances
        q = 2 * W + 1 - np.linalg.matrix_rank(Kw)

        V_cov = (Vn @ Vn.T) / q  # Covariance matrix of noise
        S_cov = (Kw.T @ np.linalg.inv(Kw @ Kw.T)) @ np.vstack([np.eye(1), np.zeros((Kw.shape[0] - 1, 1))])

        G_cov[:, :, kIdx] = np.kron(S_cov.conj().T @ S_cov, V_cov)  # Covariance matrix of G
        G_var[:, kIdx] = np.diag(G_cov[:, :, kIdx])  # Extract variance (diagonal)

    print(V_cov)

    # Reduce DFT vectors to only the identified frequencies
    R = R[:, fStartIdx:fEndIdx]
    Y = Y[:, fStartIdx:fEndIdx]

    # Loop through outputs to create dictionaries for FRD objects
    G_FRD = []
    G_var_FRD = []
    Y_contr = []

    for iOut in range(ny):
        G_FRD.append({
            'G': G[iOut, :],
            'freqVecIdent': freqVecIdent,
            'FrequencyUnit': 'Hz'
        })
        G_var_FRD.append({
            'G_var': G_var[iOut, :],
            'freqVecIdent': freqVecIdent,
            'FrequencyUnit': 'Hz'
        })
        Y_contr.append({
            'Y': Y[iOut, :],
            'GR': G[iOut, :] * R,
            'T': T[iOut, :],
            'V': V[iOut, :]
        })

    return G, G_var, Y_contr, G_cov, freqVecIdent
