"""V7 dataset with a dedicated mask for delay-aligned future controls."""

from __future__ import annotations

from .data_v5 import TCPDatasetV5


class TCPDatasetV7(TCPDatasetV5):
    def __getitem__(self, index: int):
        import torch

        item = super().__getitem__(index)
        row = self.samples.iloc[index]
        control_mask = []
        for offset in self.offsets:
            found = self.lookup.get(
                (str(row.trajectory_id), int(row.time_ms) + int(offset) * 25))
            valid = (found is not None and
                     bool(self.table.iloc[found].control_label_valid))
            control_mask.append(float(valid))
        control_mask = torch.tensor(control_mask, dtype=torch.float32)
        # NaN * 0 is still NaN. Replace unavailable targets before the masked
        # loss while retaining a separate mask so planner horizons stay intact.
        invalid = control_mask == 0
        item["future_controls"] = item["future_controls"].clone()
        item["future_controls"][invalid] = 0.0
        item["future_pedal_loss_weight"] = item["future_pedal_loss_weight"].clone()
        item["future_pedal_loss_weight"][invalid] = 1.0
        item["future_control_mask"] = control_mask
        return item
