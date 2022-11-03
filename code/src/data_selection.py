from enum import Enum


class DataSelection(Enum):
    SCENE_RELATIVE = "scene_relative"
    BODY_RELATIVE = "body_relative"
    BODY_RELATIVE_VELOCITY = "body_relative_velocity"
    BODY_RELATIVE_ACCELERATION = "body_relative_acceleration"

    def requires_relative_computation(self):
        return self == self.BODY_RELATIVE or \
               self == self.BODY_RELATIVE_VELOCITY or \
               self == self.BODY_RELATIVE_ACCELERATION

    def requires_velocity_computation(self):
        return self == self.BODY_RELATIVE_VELOCITY or \
               self == self.BODY_RELATIVE_ACCELERATION

    def requires_acceleration_computation(self):
        return self == self.BODY_RELATIVE_ACCELERATION
