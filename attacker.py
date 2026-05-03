import numpy as np

class Attacker:

    def __init__(self, simulation_parameters: dict):
        self.simulation_parameters = simulation_parameters
        self.debug = simulation_parameters["debug"]
        self.efficiency_loss = simulation_parameters["attack_properties"][
            "efficiency_loss"
        ]

    def pns_attack(self, photon_nums: np.ndarray) -> np.ndarray:

        modified_photon_nums = photon_nums.copy()

        non_vacuum_mask = photon_nums != 0

        modified_photon_nums[non_vacuum_mask] = photon_nums[non_vacuum_mask] - 1

        if self.debug:
            print(
                f"[DEBUG] Modified photon number after PNS attack:{modified_photon_nums}"
            )
            print("----------------------------------------------------------------")

        return modified_photon_nums

    def bs_attack(self, eta: float) -> float:

        modified_eta = eta * self.efficiency_loss

        if self.debug:
            print(f"[DEBUG] Modified eta after BS attack:{modified_eta}")
            print("----------------------------------------------------------------")

        return modified_eta
