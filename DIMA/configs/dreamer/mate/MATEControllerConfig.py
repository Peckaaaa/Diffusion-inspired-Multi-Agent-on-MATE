from agent.controllers.DreamerController import DreamerController
from configs.dreamer.mate.MATEAgentConfig import MATEDreamerConfig


class MATEDreamerControllerConfig(MATEDreamerConfig):
    def __init__(self):
        super().__init__()

        # Must stay 0: DreamerController's epsilon branch samples from
        # avail_actions, which MATE reports as None.
        self.epsilon = 0.
        self.EXPL_DECAY = 0.9999
        self.EXPL_NOISE = 0.
        self.EXPL_MIN = 0.

        self.temperature = 1.
        self.determinisitc = False

    def create_controller(self):
        return DreamerController(self)
