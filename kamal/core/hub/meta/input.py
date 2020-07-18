from typing import Union, Sequence

# INPUT Metadata
def ImageInput( size:          Union[Sequence[int], int],
                range:         Union[Sequence[int], int],
                space:         str,
                normalize:     Union[Sequence]=None):
    assert space in ['rgb', 'gray', 'bgr', 'rgbd']
    return dict(
        size=size, 
        range=range, 
        space=space, 
        normalize=normalize
    )