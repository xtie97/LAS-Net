# Adapted from MONAI Auto3dSeg Pipeline 


from typing import Optional, Sequence, Union

import fire

if __package__ in (None, ""):
    from segmenter import run_segmenter
else:
    from .segmenter import run_segmenter


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    override["validate#enabled"] = True
    run_segmenter(config_file=config_file, **override)


if __name__ == "__main__":
    fire.Fire()
