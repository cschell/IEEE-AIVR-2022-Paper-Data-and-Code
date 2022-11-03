from dataclasses import dataclass


@dataclass
class Metadata:
    fps = 90

    @classmethod
    def seconds2fps(cls, seconds):
        return Metadata.fps * seconds