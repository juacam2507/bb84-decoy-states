import numpy as np


class ClassicalChannel:
    def __init__(self, simulation_parameters: dict):
        """
        Initialize post-processing for decoy-state BB84 parameter estimation.

        Performs basis sifting, computes per-state gains/QBER, and extracts security
        bounds (Y₀ᴸ, Y₁ᴸ, e₁ᵘ) using two-decoy analysis. Computes asymptotic secure
        key rate via GLLP formula.

        Parameters
        ----------
        simulation_parameters : dict
            Required keys:
            - "N": int, number of pulses.
            - "debug": bool, verbose output.
            - "mu": float, signal state mean photon number.
            - "decoy_intensities": list[float], exactly 2 decoy intensities.
            - "error_correction_efficiency": float, f(EC) ∈ [1.0, 1.16].
        rng : np.random.Generator
            Random number generator for privacy amplification simulation.
        """
        self.simulation_parameters = simulation_parameters
        self.N = self.simulation_parameters["N"]
        self.debug = self.simulation_parameters["debug"]
        self.protocol = self.simulation_parameters["protocol"]
        self.signal_intensity = self.simulation_parameters["mu"]

        if self.protocol == "bb84":
            self.intensities = [self.signal_intensity]
        else:
            self.decoy_intensities = self.simulation_parameters["decoy_intensities"]
            self.intensities = np.array(
                [self.signal_intensity] + self.decoy_intensities, dtype=np.float64
            )
        self.state_num = len(self.intensities)

        self.error_correction_efficiency = self.simulation_parameters[
            "error_correction_efficiency"
        ]

    def compute_state_gain(
        self, receptor_bits: np.ndarray, state_choice: np.ndarray, state: int = 0
    ) -> float:
        """
        Compute empirical gain Qᵢ for state i.

        Gain is the sifted detection probability: :math:`Q_i = \\frac{N_{\\text{det},i}}{N_i}`

        Parameters
        ----------
        receptor_bits : npt.NDArray[np.int_]
            Bob's bits, shape (N,) ∈ {-1, 0, 1}.
        state_choice : npt.NDArray[np.int_]
            Alice's state labels, shape (N,) ∈ {0, 1, 2, ...}.
        state : int, default=0
            State index (0=signal, 1/2=decoys).

        Returns
        -------
        float
            Gain Qᵢ ∈ [0, 1].
        """

        state_mask = state_choice == state
        state_detection_mask = (receptor_bits != -1) & (state_mask)

        states_sent = np.sum(state_mask)
        states_detected = np.sum(state_detection_mask)

        if self.debug:
            print(f"[DEBUG] State {state} Detected: {states_detected}")
            print(f"[DEBUG] State {state} Sent: {states_sent}")
            print(f"[DEBUG] Gain of state {state} = {states_detected/states_sent}")
            print("----------------------------------------------------------------")

        return states_detected / states_sent

    def compute_gains(
        self, receptor_bits: np.ndarray, state_choice: np.ndarray
    ) -> np.ndarray:
        """
        Compute gains Q = [Q₀, Q₁, Q₂, ...] for all states.

        Parameters
        ----------
        receptor_bits : npt.NDArray[np.int_]
            Bob's detection outcomes, shape (N,).
        state_choice : npt.NDArray[np.int_]
            Alice's state sequence, shape (N,).

        Returns
        -------
        npt.NDArray[np.float64]
            Array of shape (state_num,) containing gains Qᵢ for each state.
        """
        gains = np.array([], dtype=float)

        for i in range(self.state_num):
            q = self.compute_state_gain(
                receptor_bits=receptor_bits, state_choice=state_choice, state=i
            )
            gains = np.append(gains, q)

        if self.debug:
            print(f"[DEBUG] Gains: {gains}")

        return gains

    def basis_reconciliation(
        self, source_basis: np.ndarray, receptor_basis: np.ndarray
    ) -> np.ndarray:
        """3*/
        Sift pulses with matching bases (Z↔Z, X↔X).

        Parameters
        ----------
        source_basis : npt.NDArray[np.int_]
            Alice's bases, shape (N,) ∈ {0, 1}.
        receptor_basis : npt.NDArray[np.int_]
            Bob's bases, shape (N,) ∈ {0, 1}.

        Returns
        -------
        npt.NDArray[np.bool_]
            Sifting mask, shape (N,) where True indicates basis match.
        """

        matching_basis_mask = source_basis == receptor_basis

        if self.debug:
            print(f"[DEBUG] Matching basis index: {matching_basis_mask}")
            print(
                f"[DEBUG] Basis coincidence rate: {np.sum(matching_basis_mask)/len(matching_basis_mask)}"
            )
            print("----------------------------------------------------------------")
        return matching_basis_mask

    def compute_state_qber(
        self,
        sifted_source_bits: np.ndarray,
        sifted_receptor_bits: np.ndarray,
        sifted_state_choice: np.ndarray,
        state: int = 0,
    ) -> float:
        """
        Compute QBER Eᵢ for state i among sifted, detected pulses.

        QBER: :math:`E_i = \\frac{\\text{Number of errors in state } i}{\\text{Number detections in state } i}`

        Parameters
        ----------
        sifted_source_bits : npt.NDArray[np.int_]
            Alice's sifted bits, shape (M,).
        sifted_receptor_bits : npt.NDArray[np.int_]
            Bob's sifted bits, shape (M,) ∈ {-1, 0, 1}.
        sifted_state_choice : npt.NDArray[np.int_]
            Sifted state labels, shape (M,).
        state : int, default=0
            State index to analyze.

        Returns
        -------
        float
            QBER Eᵢ ∈ [0, 0.5].
        """
        detected_mask = sifted_receptor_bits != -1  # True if the bit was detected
        sifted_detected_state_mask = (sifted_state_choice == state) & (
            detected_mask
        )  # True if the bit is from the state of interest and was detected
        state_error_mask = (sifted_source_bits != sifted_receptor_bits) & (
            sifted_detected_state_mask
        )  # True if the bit corresponds to the state of interest, was detected and the received bit differs from the prepared bit.

        num_state_err = np.sum(
            state_error_mask
        )  # Number of errors associated to the state (Signal or decoy)
        num_states_detected = max(
            1e-15, np.sum(sifted_detected_state_mask)
        )  # Number of detections associated to the state of interes after sifting basis

        if self.debug:
            print(
                f"[DEBUG] Non coincident detected bits associated to state {state} after sifting: {num_state_err}"
            )
            print(
                f"[DEBUG] Total detections of {state} after sifting: {num_states_detected}"
            )
            print(
                f"[DEBUG] Quantum Bit Error Rate associated to state {state}: {num_state_err/num_states_detected}"
            )
            print("----------------------------------------------------------------")
        return num_state_err / num_states_detected

    def compute_qbers(
        self,
        sifted_source_bits: np.ndarray,
        sifted_receptor_bits: np.ndarray,
        sifted_state_choice: np.ndarray,
    ) -> np.ndarray:
        """
        Compute QBERs E = [E₀, E₁, E₂, ...] for all states.

        Parameters
        ----------
        sifted_source_bits : npt.NDArray[np.int_]
            Sifted Alice bits.
        sifted_receptor_bits : npt.NDArray[np.int_]
            Sifted Bob bits.
        sifted_state_choice : npt.NDArray[np.int_]
            Sifted state labels.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of shape (state_num,) containing QBERs Eᵢ.
        """
        qbers = np.array([], dtype=float)

        for i in range(self.state_num):
            e = self.compute_state_qber(
                sifted_receptor_bits=sifted_receptor_bits,
                sifted_source_bits=sifted_source_bits,
                sifted_state_choice=sifted_state_choice,
                state=i,
            )
            qbers = np.append(qbers, e)

        if self.debug:
            print(f"[DEBUG] QBER: {qbers}")
            print("----------------------------------------------------------------")

        return qbers

    def compute_rfi_state_gain(
        self,
        receptor_bits: np.ndarray,
        basis_choice: np.ndarray,
        state_choice: np.ndarray,
        state: int = 0,
        basis: int = 0,
    ) -> float:
        """
        Compute the empirical detection gain for a selected basis and intensity/state.

        The gain is

        \[
        Q_{b,i} =
        \frac{N_{\mathrm{det},b,i}}{N_{\mathrm{sent},b,i}},
        \]

        where a detection is represented by a receptor bit different from -1.

        Parameters
        ----------
        receptor_bits : npt.NDArray[np.int_]
            Bob's outcome labels, shape (N,), with -1 indicating no detection.
        basis_choice : npt.NDArray[np.int_]
            Alice's basis choice for each emitted pulse, shape (N,).
        state_choice : npt.NDArray[np.int_]
            Alice's intensity/state label for each emitted pulse, shape (N,).
        state : int, default=0
            Intensity/state index, e.g. 0 for signal and 1/2 for decoys.
        basis : int, default=0
            Alice basis index whose gain is to be computed.

        Returns
        -------
        float
            Empirical gain Q_{basis,state}. Returns 0.0 if no pulses were sent
            with the requested basis-state combination.
        """
        if not (receptor_bits.shape == basis_choice.shape == state_choice.shape):
            raise ValueError(
                "receptor_bits, basis_choice, and state_choice must have "
                "identical shapes."
            )

        selection_mask = (basis_choice == basis) & (state_choice == state)
        detection_mask = selection_mask & (receptor_bits != -1)

        states_sent = int(np.count_nonzero(selection_mask))
        states_detected = int(np.count_nonzero(detection_mask))

        gain = states_detected / states_sent if states_sent > 0 else 0.0

        if self.debug:
            print(
                f"[DEBUG] Basis {basis}, state {state}: "
                f"detected = {states_detected}, sent = {states_sent}, "
                f"gain = {gain:.6e}"
            )
            print("----------------------------------------------------------------")

        return gain

    def compute_rfi_state_qber(
        self,
        sifted_source_bits: np.ndarray,
        sifted_receptor_bits: np.ndarray,
        sifted_basis_choice: np.ndarray,
        sifted_state_choice: np.ndarray,
        state: int = 0,
        basis: int = 0,
    ) -> float:
        """
        Compute the empirical QBER for one Alice basis and one intensity/state.

        \[
        E_{b,i} =
        \frac{N_{\mathrm{err},b,i}}{N_{\mathrm{det},b,i}}.
        \]

        A receptor bit of -1 is treated as no detection and excluded from the
        denominator and numerator.

        Parameters
        ----------
        sifted_source_bits : npt.NDArray[np.int_]
            Alice's bits after the protocol's sifting/selection step, shape (M,).
        sifted_receptor_bits : npt.NDArray[np.int_]
            Bob's corresponding outcomes, shape (M,), where -1 means no detection.
        sifted_basis_choice : npt.NDArray[np.int_]
            Alice's basis label for each retained pulse, shape (M,).
        sifted_state_choice : npt.NDArray[np.int_]
            Alice's intensity/state label for each retained pulse, shape (M,).
        state : int, default=0
            Intensity/state index, e.g. 0 for signal.
        basis : int, default=0
            Basis index to analyze.

        Returns
        -------
        float
            QBER for the selected basis-state subset. Returns 0.0 if that subset
            contains no detected pulses.
        """
        if not (
            sifted_source_bits.shape
            == sifted_receptor_bits.shape
            == sifted_basis_choice.shape
            == sifted_state_choice.shape
        ):
            raise ValueError(
                "sifted_source_bits, sifted_receptor_bits, "
                "sifted_basis_choice, and sifted_state_choice must have "
                "identical shapes."
            )

        basis_state_mask = (sifted_basis_choice == basis) & (
            sifted_state_choice == state
        )

        detected_mask = basis_state_mask & (sifted_receptor_bits != -1)

        error_mask = detected_mask & (sifted_source_bits != sifted_receptor_bits)

        num_errors = int(np.count_nonzero(error_mask))
        num_detected = int(np.count_nonzero(detected_mask))

        qber = num_errors / num_detected if num_detected > 0 else 0.0

        if self.debug:
            print(
                f"[DEBUG] Basis {basis}, state {state}: "
                f"errors = {num_errors}, detected = {num_detected}, "
                f"QBER = {qber:.6e}"
            )
            print("----------------------------------------------------------------")

        return qber

    def basis_reconciliation_rfi(
        self,
        source_basis: np.ndarray,
        receptor_basis: np.ndarray,
        z_basis: int = 0,
        x_basis: int = 1,
        y_basis: int = 2,
    ) -> dict[str, np.ndarray]:
        """
        Classify pulse indices by Alice/Bob basis pair for RFI-QKD.

        The Z-Z subset is used for raw-key generation. The X/Y basis-pair subsets
        are retained for RFI parameter estimation; they must not be discarded
        merely because Alice and Bob selected different transverse basis labels.

        Parameters
        ----------
        source_basis : npt.NDArray[np.int_]
            Alice's basis labels, shape (N,). Expected values: {z_basis, x_basis,
            y_basis}.
        receptor_basis : npt.NDArray[np.int_]
            Bob's basis labels, shape (N,). Expected values: {z_basis, x_basis,
            y_basis}.
        z_basis : int, default=0
            Label used for the shared key basis Z.
        x_basis : int, default=1
            Label used for the transverse X basis.
        y_basis : int, default=2
            Label used for the transverse Y basis.

        Returns
        -------
        dict[str, npt.NDArray[np.bool_]]
            Boolean masks, each with shape (N,):

            - ``"ZZ"``: key-generation events
            - ``"XX"``, ``"XY"``, ``"YX"``, ``"YY"``: RFI parameter-estimation
            events
            - ``"transverse"``: union of the four X/Y masks
            - ``"retained"``: union of ZZ and transverse events
        """
        if source_basis.shape != receptor_basis.shape:
            raise ValueError(
                "source_basis and receptor_basis must have identical shapes."
            )

        zz_mask = (source_basis == z_basis) & (receptor_basis == z_basis)
        xx_mask = (source_basis == x_basis) & (receptor_basis == x_basis)
        xy_mask = (source_basis == x_basis) & (receptor_basis == y_basis)
        yx_mask = (source_basis == y_basis) & (receptor_basis == x_basis)
        yy_mask = (source_basis == y_basis) & (receptor_basis == y_basis)

        transverse_mask = xx_mask | xy_mask | yx_mask | yy_mask
        retained_mask = zz_mask | transverse_mask

        masks = {
            "ZZ": zz_mask,
            "XX": xx_mask,
            "XY": xy_mask,
            "YX": yx_mask,
            "YY": yy_mask,
            "transverse": transverse_mask,
            "retained": retained_mask,
        }

        if self.debug:
            total = len(source_basis)
            print("[DEBUG] RFI basis-pair counts:")
            for label in ("ZZ", "XX", "XY", "YX", "YY"):
                count = np.count_nonzero(masks[label])
                print(f"  {label}: {count} ({count / total:.4%})")

            retained = np.count_nonzero(retained_mask)
            print(f"[DEBUG] Retained RFI events: {retained} ({retained / total:.4%})")
            print("----------------------------------------------------------------")

        return masks

    def compute_gains_rfi(
        self,
        receptor_bits: np.ndarray,
        source_basis: np.ndarray,
        receptor_basis: np.ndarray,
        state_choice: np.ndarray,
        basis_num: int = 3,
    ) -> np.ndarray:
        """
        Compute gain Q_{a,b,i} for every Alice/Bob basis pair and state.

        \[
        Q_{a,b,i} =
        \frac{N_{\mathrm{det},a,b,i}}{N_{\mathrm{sent},a,b,i}}.
        \]

        Parameters
        ----------
        receptor_bits : npt.NDArray[np.int_]
            Bob's outcomes, shape (N,), where -1 denotes no detection.
        source_basis : npt.NDArray[np.int_]
            Alice's preparation-basis label, shape (N,).
        receptor_basis : npt.NDArray[np.int_]
            Bob's measurement-basis label, shape (N,).
        state_choice : npt.NDArray[np.int_]
            Alice's intensity/state label, shape (N,).
        basis_num : int, default=3
            Number of bases. Use 3 for an RFI protocol with Z, X, and Y.

        Returns
        -------
        npt.NDArray[np.float64]
            Array ``gains`` with shape ``(basis_num, basis_num, state_num)``,
            where ``gains[a, b, i] = Q_{a,b,i}``.

            The first index is Alice's basis, the second is Bob's basis, and the
            third is the signal/decoy state index.
        """
        if not (
            receptor_bits.shape
            == source_basis.shape
            == receptor_basis.shape
            == state_choice.shape
        ):
            raise ValueError(
                "receptor_bits, source_basis, receptor_basis, and state_choice "
                "must have identical shapes."
            )

        gains = np.zeros((basis_num, basis_num, self.state_num), dtype=float)

        for alice_basis in range(basis_num):
            for bob_basis in range(basis_num):
                basis_pair_mask = (source_basis == alice_basis) & (
                    receptor_basis == bob_basis
                )

                for state in range(self.state_num):
                    selection_mask = basis_pair_mask & (state_choice == state)

                    num_sent = np.count_nonzero(selection_mask)
                    num_detected = np.count_nonzero(
                        selection_mask & (receptor_bits != -1)
                    )

                    gains[alice_basis, bob_basis, state] = (
                        num_detected / num_sent if num_sent > 0 else 0.0
                    )

        if self.debug:
            basis_names = ("Z", "X", "Y")

            for alice_basis in range(basis_num):
                for bob_basis in range(basis_num):
                    a_name = basis_names[alice_basis]
                    b_name = basis_names[bob_basis]
                    print(
                        f"[DEBUG] Gains {a_name}{b_name}: "
                        f"{gains[alice_basis, bob_basis]}"
                    )

            print("----------------------------------------------------------------")

        return gains

    def compute_qbers_by_basis(
        self,
        sifted_source_bits: np.ndarray,
        sifted_receptor_bits: np.ndarray,
        sifted_source_basis: np.ndarray,
        sifted_receptor_basis: np.ndarray,
        sifted_state_choice: np.ndarray,
        basis_num: int = 3,
    ) -> np.ndarray:
        """
        Compute QBER E_{a,b,i} for every Alice/Bob basis pair and state.

        \[
        E_{a,b,i} =
        \frac{N_{\mathrm{err},a,b,i}}{N_{\mathrm{det},a,b,i}}.
        \]

        Parameters
        ----------
        sifted_source_bits : npt.NDArray[np.int_]
            Alice's bit values, shape (M,).
        sifted_receptor_bits : npt.NDArray[np.int_]
            Bob's outcomes, shape (M,), where -1 denotes no detection.
        sifted_source_basis : npt.NDArray[np.int_]
            Alice's preparation-basis labels, shape (M,).
        sifted_receptor_basis : npt.NDArray[np.int_]
            Bob's measurement-basis labels, shape (M,).
        sifted_state_choice : npt.NDArray[np.int_]
            Alice's intensity/state labels, shape (M,).
        basis_num : int, default=3
            Number of basis labels. Use 3 for Z, X, and Y RFI-QKD.

        Returns
        -------
        npt.NDArray[np.float64]
            ``qbers`` with shape ``(basis_num, basis_num, self.state_num)``.

            ``qbers[a, b, i]`` is the QBER for Alice basis ``a``, Bob basis ``b``,
            and state/intensity ``i``. A value of 0.0 is returned when the selected
            subset has no detected pulses.
        """
        if not (
            sifted_source_bits.shape
            == sifted_receptor_bits.shape
            == sifted_source_basis.shape
            == sifted_receptor_basis.shape
            == sifted_state_choice.shape
        ):
            raise ValueError(
                "All sifted bit, basis, and state arrays must have identical " "shapes."
            )

        qbers = np.zeros(
            (basis_num, basis_num, self.state_num),
            dtype=float,
        )

        detected_mask = sifted_receptor_bits != -1

        for alice_basis in range(basis_num):
            for bob_basis in range(basis_num):
                basis_pair_mask = (sifted_source_basis == alice_basis) & (
                    sifted_receptor_basis == bob_basis
                )

                for state in range(self.state_num):
                    selected_detected_mask = (
                        basis_pair_mask & (sifted_state_choice == state) & detected_mask
                    )

                    error_mask = selected_detected_mask & (
                        sifted_source_bits != sifted_receptor_bits
                    )

                    num_detected = np.count_nonzero(selected_detected_mask)
                    num_errors = np.count_nonzero(error_mask)

                    qbers[alice_basis, bob_basis, state] = (
                        num_errors / num_detected if num_detected > 0 else 0.0
                    )

        if self.debug:
            basis_names = ("Z", "X", "Y")

            print("[DEBUG] QBERs by Alice/Bob basis pair and state:")
            for alice_basis in range(basis_num):
                for bob_basis in range(basis_num):
                    print(
                        f"  E_{basis_names[alice_basis]}"
                        f"{basis_names[bob_basis]}: "
                        f"{qbers[alice_basis, bob_basis]}"
                    )

            print("----------------------------------------------------------------")

        return qbers
