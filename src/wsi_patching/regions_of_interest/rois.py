from dataclasses import dataclass
from typing import List, Tuple

Box = Tuple[int, int, int, int]


class ROI:
    def bounds(self) -> Box:
        raise NotImplementedError

    def contains_point(self, x: float, y: float) -> bool:
        raise NotImplementedError

    def contains_patch(self, bx, by, bw, bh) -> bool:
        return (
            self.contains_point(bx, by)
            and self.contains_point(bx + bw - 1, by)
            and self.contains_point(bx, by + bh - 1)
            and self.contains_point(bx + bw - 1, by + bh - 1)
        )

    def intersects_patch(self, bx, by, bw, bh) -> bool:
        return (
            self.contains_point(bx, by)
            or self.contains_point(bx + bw - 1, by)
            or self.contains_point(bx, by + bh - 1)
            or self.contains_point(bx + bw - 1, by + bh - 1)
        )


@dataclass
class BoxROI(ROI):
    x: int
    y: int
    w: int
    h: int

    def bounds(self) -> Box:
        return (self.x, self.y, self.w, self.h)

    def contains_point(self, x: float, y: float) -> bool:
        return (self.x <= x < self.x + self.w) and (self.y <= y < self.y + self.h)


@dataclass
class RectAreaROI:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def subdivide(self, max_size: int, tile_size: int, stride: int) -> List["RectAreaROI"]:
        """
        Split this rectangle into smaller rectangles if width or height > max_size.
        Splits are aligned to stride so that tiles stay consistent.
        """
        sub_rois: List[RectAreaROI] = []
        x_end = self.x + self.w
        y_end = self.y + self.h

        for yy in range(self.y, y_end, max_size):
            for xx in range(self.x, x_end, max_size):
                ww = min(max_size, x_end - xx)
                hh = min(max_size, y_end - yy)

                # Align to stride boundaries: ensure splits produce valid tile starts
                # (optional: round ww/hh up so they cover full tiles)
                aligned_w = (ww // stride) * stride
                aligned_h = (hh // stride) * stride
                if aligned_w < tile_size or aligned_h < tile_size:
                    continue

                sub_rois.append(RectAreaROI(xx, yy, aligned_w, aligned_h))

        return sub_rois
