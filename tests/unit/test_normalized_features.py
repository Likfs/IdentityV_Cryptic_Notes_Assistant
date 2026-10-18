"""Downsampling must not discard short walls solely because pixel units changed."""
import cv2
import numpy as np
from src.structure_layout.features import target_wall_edges


def test_short_source_wall_survives_recognition_downsampling():
    source=np.full((128,128,3),100,np.uint8)
    cv2.line(source,(64,40),(64,55),(175,175,175),1)
    small=cv2.resize(source,(64,64),interpolation=cv2.INTER_AREA)
    assert target_wall_edges(source)[0].sum()>0
    # Keeping original pixel lengths incorrectly erases this valid wall.
    assert target_wall_edges(small)[0].sum()==0
    edges=target_wall_edges(small,pixel_scale=.5)
    assert edges[0,19:29,30:35].sum()>0
    assert not edges[1].any()
