from typing import Iterator

import torch.nn as nn

class MultiTaskModel:
    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters shared by all tasks.
        Returns
        -------
        """
        return NotImplemented

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters specific to each task.
        Returns
        -------
        """
        return NotImplemented

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        return NotImplemented