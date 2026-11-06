"""Supported asset suffixes in display order; only actual images declare a floor."""

FLOOR_NAMES = {
    '3rd': '3 楼', '2nd': '2 楼', '1st': '1 楼',
    'b1': '地下1层', 'b2': '地下2层', 'b3': '地下3层', 'b4': '地下4层',
}
FLOORS = tuple(FLOOR_NAMES)
# Preserve the interpretation of existing callers/configurations that omit floors.
DEFAULT_FLOORS = ('2nd', '1st', 'b1')
