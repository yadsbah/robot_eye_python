import cv2
import numpy as np
import math
import time
import requests
import json
import os
import websocket
from collections import deque

# ===================== CAMERA / NETWORK CONFIG =====================
CAMERA_IP                  = "192.168.0.82"
STREAM_URL                 = f"http://{CAMERA_IP}:81/stream"
LINE_SENSOR_WS_URL         = "ws://localhost:3002"
LINE_SENSOR_INTERVAL_MS    = 20   # send line data every N milliseconds
CAMERA_READ_FAILURE_LIMIT  = 15   # reconnect after this many bad reads
CAMERA_RECONNECT_DELAY_SEC = 1.0  # short pause before reopening stream
# ==================================================================

# ===================== PATH DETECTION TUNING ======================
# --- ROI ---
# Fraction of frame height from the TOP where the floor ROI starts.
# 0.5 = bottom half; 0.4 = bottom 60 %.  Start here, adjust if the
# path is cut off or too much non-floor content is included.
PATH_ROI_TOP_FRACTION      = 0.77

# --- Blur ---
# Median kernel (must be odd).  Applied FIRST to kill salt-and-pepper
# noise (speckled floors with white/black dots).  Set to 1 to skip.
PATH_MEDIAN_KERNEL         = 9

# Gaussian kernel (must be odd).  Applied AFTER median to smooth
# remaining texture before thresholding.
PATH_BLUR_KERNEL           = 9

# --- Threshold / binary path mask ---
# The path detector now uses one binary thresholding path only:
# threshold the ROI to black/white, then process only that mask.
# White pixels = candidate path, black pixels = background.
PATH_BINARY_THRESHOLD      = 120

# Set True when the path is DARKER than the background (black line on
# light floor).  Inverts the binary mask so the dark path becomes white.
PATH_INVERT_THRESHOLD      = False

# Bottom selection band used to decide which white connected component
# is the robot's current path.  Any component that touches this bottom
# band can be selected, including paths near the left/right corners.
PATH_MAIN_SELECTION_ROWS   = (0.82, 1.00)

# Small centre preference when multiple components touch the bottom.
# Keep this low so edge/corner paths are still accepted.
PATH_MAIN_CENTER_BIAS      = 0.15

# --- Morphology ---
# Kernel size for open (noise removal) then close (gap filling).
# Increase if the mask is fragmented; decrease if fine details matter.
PATH_MORPH_KERNEL_SIZE     = 9

# --- Contour filter ---
# Contours smaller than this (px²) are ignored as noise.
PATH_MIN_CONTOUR_AREA      = 2000

# A valid path contour must reach the bottom N% of the ROI.
# This rejects noise blobs floating in the middle of the image.
# Lower (e.g. 0.70) if the robot sees the path far ahead only.
PATH_CONTOUR_MUST_REACH_BOTTOM = 0.85

# Minimum solidity (area / convex-hull area).  Real lines are fairly
# solid (>0.3); scattered dot clusters have low solidity (<0.2).
PATH_MIN_SOLIDITY          = 0.25

# Minimum aspect ratio of the bounding rect (long / short side).
# Lines are elongated (>1.5); round noise blobs are near 1.0.
# Set to 1.0 to disable this check.
PATH_MIN_ASPECT_RATIO      = 1.3

# --- Centerline scanlines ---
# Number of horizontal rows sampled from bottom upward.
# More lines → smoother heading estimate but negligible extra cost.
PATH_SCANLINE_COUNT        = 20

# Ignore unusually wide scanlines when computing off-center error.
# Wide rows often happen at corners/intersections and can make the
# midpoint look falsely close to the image centre.
PATH_ERROR_MAX_WIDTH_RATIO = 1.35

# --- Multi-path / secondary contours ---
# Secondary contours must be at least this fraction of the primary
# contour area to be drawn (prevents tiny noise patches appearing).
PATH_SECONDARY_MIN_AREA_RATIO  = 0.15
# If a scanline row's path width exceeds median_width × this factor,
# the row is flagged as a potential junction / intersection.
PATH_INTERSECTION_WIDTH_RATIO  = 1.8

# --- Branch / intersection direction detection ---
# A direction is only counted when the path REACHES the corresponding
# edge of the ROI.  We sample a thin strip right at each boundary.
#
# PATH_BRANCH_EDGE_THICKNESS – strip depth as a fraction of the ROI
#   dimension (height for forward/back, width for left/right).
#   e.g. 0.06 = top/bottom 6 % of ROI height, left/right 6 % of width.
#   Increase if the camera is close to the floor and edges are noisy.
PATH_BRANCH_EDGE_THICKNESS = 0.06

# Fraction of the edge strip that must be white to count as a path.
# Raise to 0.15+ if open floor reflections trigger false detections.
# Lower to 0.05 if a real path is being missed at the edge.
PATH_BRANCH_EDGE_MIN_FILL  = 0.10

# Centre column fraction checked for the FORWARD and BACKWARD edges.
# Narrow this if walls or furniture at the sides trigger false FWD.
PATH_BRANCH_FWD_COLS   = (0.25, 0.75)
PATH_BRANCH_BACK_COLS  = (0.25, 0.75)

# Row fraction checked for the LEFT and RIGHT edges.
# Keep this wide so a narrow side-path anywhere along the wall is caught.
PATH_BRANCH_SIDE_ROWS  = (0.05, 0.95)

# --- Temporal smoothing ---
# Exponential moving average factor for the lateral error.
# Lower → smoother but more lag.  Range 0..1.
PATH_SMOOTHING_ALPHA       = 0.75

# --- Debug output ---
PATH_SHOW_DEBUG            = True   # set False for headless deployment
PATH_SAVE_DEBUG_EVERY_N    = 0      # save every N frames (0 = off)
PATH_DEBUG_SAVE_DIR        = "debug_frames"

# --- Optional bird's-eye / perspective warp ---
# Enable only when the camera is at a fixed, known tilt angle.
# Tune the four SRC corners until the floor looks rectangular in the warp.
PATH_ENABLE_BIRDSEYE       = False
# Source quad in the ROI expressed as (x_frac, y_frac) of ROI size.
# Order: top-left, top-right, bottom-right, bottom-left.
PATH_BIRDSEYE_SRC          = np.float32([[0.20, 0.0],
                                          [0.80, 0.0],
                                          [1.00, 1.0],
                                          [0.00, 1.0]])
PATH_BIRDSEYE_DST          = np.float32([[0.0, 0.0],
                                          [1.0, 0.0],
                                          [1.0, 1.0],
                                          [0.0, 1.0]])
# ==================================================================


# ──────────────────────────────────────────────────────────────────
#  PATH DETECTOR
# ──────────────────────────────────────────────────────────────────

class PathDetector:
    """
    Classical OpenCV pipeline for bright-floor path detection.

    Call detect_path(frame) every frame.  The returned dict contains:

        mask           – binary mask of the segmented path (ROI dimensions)
        contour        – chosen path contour in ROI coordinates (or None)
        center_points  – list of (x, y) midpoints in full-frame coordinates
        lateral_error  – signed px offset of path centre from image centre
                         positive → path is to the RIGHT of centre
        heading_deg    – estimated path heading (degrees from vertical,
                         positive → path leans right); None if unavailable
        debug_frame    – black/white processing view plus dashboard
    """

    def __init__(self):
        k = PATH_MORPH_KERNEL_SIZE
        self._morph_kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._smoothed_error  = None   # populated after first valid detection
        self._frame_count     = 0

        if PATH_SAVE_DEBUG_EVERY_N > 0:
            os.makedirs(PATH_DEBUG_SAVE_DIR, exist_ok=True)

    # ── public ─────────────────────────────────────────────────────

    def detect_path(self, frame: np.ndarray) -> dict:
        """Run the full pipeline and return the result dict."""
        self._frame_count += 1
        h, w = frame.shape[:2]

        # ── Step 1: grayscale ──────────────────────────────────────
        gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3 else frame.copy())

        # ── Step 2: crop ROI to lower portion of the frame ─────────
        roi_y0 = int(h * PATH_ROI_TOP_FRACTION)
        roi    = gray[roi_y0:h, 0:w]

        # ── Step 3: optional bird's-eye warp ──────────────────────
        if PATH_ENABLE_BIRDSEYE:
            roi = self._warp_birdseye(roi)

        roi_h, roi_w = roi.shape[:2]

        # ── Step 4a: Median blur (kills salt-and-pepper speckle) ────
        if PATH_MEDIAN_KERNEL > 1:
            roi = cv2.medianBlur(roi, PATH_MEDIAN_KERNEL)

        # ── Step 4b: Gaussian blur (smooths remaining texture) ────
        blurred = cv2.GaussianBlur(
            roi, (PATH_BLUR_KERNEL, PATH_BLUR_KERNEL), sigmaX=0)

        # ── Step 5: threshold – convert the ROI to black/white ─────
        binary_mask = self._threshold(blurred)

        # ── Step 6: clean the binary image and keep only the current
        #           bottom-connected path component ──────────────────
        mask = self._build_path_mask(binary_mask, roi_w, roi_h)

        # ── Step 7: find external contours ─────────────────────────
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ── Step 8: classify contours into primary + secondaries ───
        best, secondary = self._classify_contours(cnts, roi_w, roi_h)

        if best is None:
            # Fallback: still estimate the error from the filtered mask
            # so the off-center value keeps updating even when contour
            # classification temporarily rejects the path.
            center_pts_roi, raw_error, heading_deg, _ = \
                self._extract_centerline(mask, roi_w, roi_h)
            center_pts_global = [(cx, cy + roi_y0) for cx, cy in center_pts_roi]

            if center_pts_roi:
                if self._smoothed_error is None:
                    self._smoothed_error = raw_error
                else:
                    self._smoothed_error = (
                        PATH_SMOOTHING_ALPHA * raw_error
                        + (1.0 - PATH_SMOOTHING_ALPHA) * self._smoothed_error)

            fallback_err = self._smoothed_error if self._smoothed_error is not None else 0.0
            directions   = self._detect_branches(mask, roi_w, roi_h)
            cam_overlay  = self._draw_process_view(
                frame, roi_y0, roi_w, roi_h, mask, None, [], center_pts_global, [])
            dashboard    = self._draw_dashboard(
                w, fallback_err, heading_deg if center_pts_roi else None, directions, fps=0.0)
            result = self._empty_result(frame, mask, fallback_err)
            result["center_points"] = center_pts_global
            result["heading_deg"] = heading_deg if center_pts_roi else None
            result["directions"]  = directions
            result["debug_frame"] = np.vstack([cam_overlay, dashboard])
            self._maybe_save(result["debug_frame"])
            return result

        # ── Step 9: rasterise primary contour for scanning ─────────
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [best], -1, 255, cv2.FILLED)

        # ── Step 10: centerline + intersection detection (primary) ──
        center_pts_roi, raw_error, heading_deg, inter_pts_roi = \
            self._extract_centerline(contour_mask, roi_w, roi_h)

        # Translate primary results from ROI-local to full-frame coords.
        center_pts_global = [(cx, cy + roi_y0) for cx, cy in center_pts_roi]
        inter_pts_global  = [(cx, cy + roi_y0) for cx, cy in inter_pts_roi]

        # ── Step 10b: centerlines for every secondary contour ──────
        secondary_centerlines = []   # list of lists of (x, y) global pts
        for sec_cnt in secondary:
            sec_mask = np.zeros_like(mask)
            cv2.drawContours(sec_mask, [sec_cnt], -1, 255, cv2.FILLED)
            sec_pts, _, _, sec_inter = self._extract_centerline(
                sec_mask, roi_w, roi_h)
            secondary_centerlines.append(
                [(cx, cy + roi_y0) for cx, cy in sec_pts])
            # Merge secondary intersection points into the global list.
            inter_pts_global.extend(
                [(cx, cy + roi_y0) for cx, cy in sec_inter])

        # ── Step 11: temporal smoothing of lateral error ───────────
        if self._smoothed_error is None:
            self._smoothed_error = raw_error
        else:
            self._smoothed_error = (
                PATH_SMOOTHING_ALPHA * raw_error
                + (1.0 - PATH_SMOOTHING_ALPHA) * self._smoothed_error)

        directions = self._detect_branches(mask, roi_w, roi_h)

        cam_overlay = self._draw_process_view(
            frame, roi_y0, roi_w, roi_h,
            mask, best, secondary,
            center_pts_global, secondary_centerlines)

        dashboard = self._draw_dashboard(
            w, self._smoothed_error, heading_deg, directions, fps=0.0)

        debug = np.vstack([cam_overlay, dashboard])

        result = {
            "mask":                  mask,
            "contour":               best,
            "secondary_contours":    secondary,
            "center_points":         center_pts_global,
            "secondary_centerlines": secondary_centerlines,
            "intersection_points":   inter_pts_global,
            "lateral_error":         self._smoothed_error,
            "heading_deg":           heading_deg,
            "directions":            directions,
            "debug_frame":           debug,
        }

        # ── Step 12: optional periodic save ────────────────────────
        self._maybe_save(debug)
        return result

    # ── private helpers ────────────────────────────────────────────

    def _threshold(self, gray_roi: np.ndarray) -> np.ndarray:
        flags = cv2.THRESH_BINARY_INV if PATH_INVERT_THRESHOLD else cv2.THRESH_BINARY
        _, mask = cv2.threshold(
            gray_roi, PATH_BINARY_THRESHOLD, 255, flags)
        return mask

    def _build_path_mask(self, binary_mask: np.ndarray,
                         roi_w: int, roi_h: int) -> np.ndarray:
        """
        Starting from a black/white threshold image, keep only the white
        component that the robot is actually standing on.
        """
        mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, self._morph_kernel, iterations=2)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self._morph_kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= PATH_MIN_CONTOUR_AREA:
                clean_mask[labels == lbl] = 255

        return self._extract_main_path_mask(clean_mask, roi_w, roi_h)

    @staticmethod
    def _extract_main_path_mask(mask: np.ndarray,
                                roi_w: int, roi_h: int) -> np.ndarray:
        """
        From the thresholded black/white mask, keep the connected white
        component that most strongly connects to the bottom of the ROI.
        This still works when the path runs near the left or right edge.
        """
        if mask.size == 0 or not np.any(mask):
            return np.zeros_like(mask)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        band_r0 = int(PATH_MAIN_SELECTION_ROWS[0] * roi_h)
        band_r1 = min(roi_h, max(band_r0 + 1,
                                 int(PATH_MAIN_SELECTION_ROWS[1] * roi_h)))
        bottom_band = labels[band_r0:band_r1, :]

        best_label = 0
        best_score = -1.0
        cx_frame = roi_w / 2.0

        for lbl in range(1, num_labels):
            bottom_pixels = int(np.count_nonzero(bottom_band == lbl))
            if bottom_pixels == 0:
                continue

            area = float(stats[lbl, cv2.CC_STAT_AREA])
            left = float(stats[lbl, cv2.CC_STAT_LEFT])
            width = float(stats[lbl, cv2.CC_STAT_WIDTH])
            centroid_x = left + width / 2.0
            centre_bonus = 1.0 - (
                PATH_MAIN_CENTER_BIAS * abs(centroid_x - cx_frame) / max(cx_frame, 1.0)
            )

            # Prefer components with strong contact at the bottom of the
            # image, while keeping only a gentle centre preference.
            score = (bottom_pixels * 4.0 + area * 0.02) * centre_bonus
            if score > best_score:
                best_score = score
                best_label = lbl

        if best_label == 0:
            return np.zeros_like(mask)

        return np.where(labels == best_label, np.uint8(255), np.uint8(0))

    def _classify_contours(self, contours, roi_w: int, roi_h: int):
        """
        Partition all contours into (best, secondary_list).

        Each contour must pass:
          1. Minimum area
          2. Must reach the bottom portion of the ROI
          3. Minimum solidity (filters scattered dot clusters)
          4. Minimum aspect ratio (filters round noise blobs)

        Scoring: area × centre_penalty, where the penalty gently
        discounts contours whose centroid is far from the frame centre.
        """
        if not contours:
            return None, []

        cx_frame   = roi_w / 2.0
        bottom_thr = int(roi_h * PATH_CONTOUR_MUST_REACH_BOTTOM)
        scored     = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < PATH_MIN_CONTOUR_AREA:
                continue

            # Bottom-reach check: the contour's lowest point must be
            # below bottom_thr (i.e. it touches or nearly touches the
            # bottom of the ROI — where the robot actually is).
            max_y = cnt[:, :, 1].max()
            if max_y < bottom_thr:
                continue

            # Solidity: reject fragmented / scattered noise clusters.
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < PATH_MIN_SOLIDITY:
                    continue

            # Aspect ratio: reject round blobs that are not line-shaped.
            _, (bw, bh), _ = cv2.minAreaRect(cnt)
            if min(bw, bh) > 0:
                aspect = max(bw, bh) / min(bw, bh)
                if aspect < PATH_MIN_ASPECT_RATIO:
                    continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            centre_penalty = 1.0 - 0.5 * abs(cx - cx_frame) / cx_frame
            scored.append((area * centre_penalty, area, cnt))

        if not scored:
            return None, []

        scored.sort(key=lambda t: t[0], reverse=True)
        best_area = scored[0][1]
        best      = scored[0][2]

        min_secondary_area = best_area * PATH_SECONDARY_MIN_AREA_RATIO
        secondary = [cnt for (_, area, cnt) in scored[1:]
                     if area >= min_secondary_area]

        return best, secondary

    def _extract_centerline(self, contour_mask: np.ndarray,
                             roi_w: int, roi_h: int):
        """
        Scan PATH_SCANLINE_COUNT evenly spaced rows from the bottom of the
        contour mask upward.  For each row find the leftmost and rightmost
        white pixel and record their midpoint and width.

        Also flags rows where the width is much wider than the median as
        potential junctions / intersections.

        Returns (center_points, lateral_error, heading_deg, intersection_pts).
          center_points    – list of (x, y) midpoints in ROI coordinates
          lateral_error    – bottom midpoint x minus frame centre x
          heading_deg      – path angle from vertical (+ = leaning right)
          intersection_pts – list of (x, y) midpoints at junction rows
        """
        center_points = []
        widths        = []
        spacing = max(1, roi_h // (PATH_SCANLINE_COUNT + 1))

        for i in range(1, PATH_SCANLINE_COUNT + 1):
            row_y = roi_h - i * spacing   # scan upward from the bottom
            if row_y < 0:
                break

            cols = np.where(contour_mask[row_y] > 0)[0]
            if cols.size == 0:
                continue

            left  = int(cols[0])
            right = int(cols[-1])
            mid_x = (left + right) // 2
            width = right - left + 1
            center_points.append((mid_x, row_y))
            widths.append(width)

        if not center_points:
            return [], 0.0, 0.0, []

        # Lateral error should use the most trustworthy lower scanlines.
        # Very wide rows often happen at turns/corners and make the path
        # look more centered than it really is.
        median_w = float(np.median(widths))
        max_ok_w = median_w * PATH_ERROR_MAX_WIDTH_RATIO
        reliable_points = [(pt, w) for pt, w in zip(center_points, widths)
                           if w <= max_ok_w]
        if not reliable_points:
            reliable_points = list(zip(center_points, widths))

        lateral_error = 0.0
        if len(reliable_points) >= 2:
            xs = np.array([pt[0] for pt, _ in reliable_points], dtype=np.float32)
            ys = np.array([pt[1] for pt, _ in reliable_points], dtype=np.float32)
            coeffs = np.polyfit(ys, xs, 1)
            bottom_est_x = float(np.polyval(coeffs, roi_h - 1))
            lateral_error = bottom_est_x - roi_w / 2.0
        else:
            bottom_cx, _ = reliable_points[0][0]
            lateral_error = float(bottom_cx - roi_w / 2.0)

        # Fit a line through all centre points to estimate heading.
        heading_deg = 0.0
        if len(center_points) >= 2:
            xs = np.array([p[0] for p in center_points], dtype=np.float32)
            ys = np.array([p[1] for p in center_points], dtype=np.float32)
            # Regress x on y (rows are independent variable).
            # slope = dx/dy; atan gives angle of path from vertical.
            coeffs      = np.polyfit(ys, xs, 1)
            heading_deg = float(math.degrees(math.atan(coeffs[0])))

        # Intersection / junction detection: rows that are significantly
        # wider than the median width are likely branch or crossing points.
        intersection_pts = []
        if widths:
            threshold = median_w * PATH_INTERSECTION_WIDTH_RATIO
            for (pt, w) in zip(center_points, widths):
                if w > threshold:
                    intersection_pts.append(pt)

        return center_points, lateral_error, heading_deg, intersection_pts

    def _warp_birdseye(self, roi: np.ndarray) -> np.ndarray:
        """Perspective-warp the ROI to a bird's-eye view."""
        rh, rw = roi.shape[:2]
        src = (PATH_BIRDSEYE_SRC * np.array([rw, rh], dtype=np.float32))
        dst = (PATH_BIRDSEYE_DST * np.array([rw, rh], dtype=np.float32))
        M   = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(roi, M, (rw, rh))

    # ── visualization ─────────────────────────────────────────────

    def _draw_process_view(self, frame, roi_y0, roi_w, roi_h,
                           mask, contour, secondary_contours,
                           center_points, secondary_centerlines) -> np.ndarray:
        """
        Render the normal camera frame, but replace only the lower ROI
        with the processed black/white path view.
        """
        dbg = frame.copy()
        _, w = dbg.shape[:2]

        # ROI boundary
        cv2.line(dbg, (0, roi_y0), (w, roi_y0), (90, 90, 90), 1)

        # Show the processed ROI directly in black and white.
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        dbg[roi_y0:roi_y0 + roi_h, 0:roi_w] = mask_bgr

        # Secondary path contours
        for sec_cnt in (secondary_contours or []):
            shifted = sec_cnt + np.array([[[0, roi_y0]]])
            cv2.drawContours(dbg, [shifted], -1, (180, 180, 180), 1)

        # Primary path contour
        if contour is not None:
            shifted = contour + np.array([[[0, roi_y0]]])
            cv2.drawContours(dbg, [shifted], -1, (255, 255, 255), 2)

        # Secondary centerlines
        for sec_pts in (secondary_centerlines or []):
            if not sec_pts:
                continue
            pts = np.array(sec_pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(dbg, [pts], isClosed=False,
                          color=(160, 160, 160), thickness=1)
            for cx, cy in sec_pts:
                cv2.circle(dbg, (cx, cy), 3, (160, 160, 160), -1)

        # Primary centerline
        if center_points:
            pts = np.array(center_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(dbg, [pts], isClosed=False,
                          color=(220, 220, 220), thickness=2)
            for cx, cy in center_points:
                cv2.circle(dbg, (cx, cy), 4, (220, 220, 220), -1)

        return dbg

    @staticmethod
    def _detect_branches(mask: np.ndarray, roi_w: int, roi_h: int) -> dict:
        """
        Direction detection with connectivity enforcement.

        Step 1 – find the connected component the robot is currently on
                 by probing a small rectangle at the bottom-centre of the
                 ROI (where the robot sits in the image).
        Step 2 – build a single-component mask from that label only.
        Step 3 – run edge-strip checks on that mask so only paths that
                 are physically joined to the robot's current path can
                 activate a direction arrow.

        Any disconnected noise, furniture, or reflections at the image
        edge are automatically excluded because they belong to a
        different connected component.
        """
        no_path = {"forward": False, "backward": False,
                   "left": False, "right": False}

        # ── Step 1: isolate the bottom-centre-connected path ──────
        main_mask = PathDetector._extract_main_path_mask(mask, roi_w, roi_h)
        if not np.any(main_mask):
            return no_path   # robot is not on any path right now

        # ── Step 2: edge-strip checks on the connected mask ───────
        def strip_hit(strip: np.ndarray) -> bool:
            if strip.size == 0:
                return False
            return (np.count_nonzero(strip) / strip.size) >= PATH_BRANCH_EDGE_MIN_FILL

        er  = max(4, int(PATH_BRANCH_EDGE_THICKNESS * roi_h))
        ec  = max(4, int(PATH_BRANCH_EDGE_THICKNESS * roi_w))

        fc0 = int(PATH_BRANCH_FWD_COLS[0]  * roi_w)
        fc1 = max(fc0 + 1, int(PATH_BRANCH_FWD_COLS[1]  * roi_w))
        bc0 = int(PATH_BRANCH_BACK_COLS[0] * roi_w)
        bc1 = max(bc0 + 1, int(PATH_BRANCH_BACK_COLS[1] * roi_w))
        sr0 = int(PATH_BRANCH_SIDE_ROWS[0] * roi_h)
        sr1 = max(sr0 + 1, int(PATH_BRANCH_SIDE_ROWS[1] * roi_h))

        return {
            "forward":  strip_hit(main_mask[0:er,              fc0:fc1]),
            "backward": strip_hit(main_mask[roi_h - er:roi_h,  bc0:bc1]),
            "left":     strip_hit(main_mask[sr0:sr1,           0:ec]),
            "right":    strip_hit(main_mask[sr0:sr1,           roi_w - ec:roi_w]),
        }

    @staticmethod
    def _draw_dashboard(w: int, lateral_error: float, heading_deg,
                        directions: dict, fps: float) -> np.ndarray:
        """
        Build a 160-px-tall instrument panel to stack below the camera frame.

        ┌──────────────────────────────────────────────────────────────┐
        │  OFF CENTER  [════════════|●══════|═══════════]  +23px →    │
        ├──────────────────────────────────────────────────────────────┤
        │          ▲ FORWARD                                           │
        │  ◄ LEFT              RIGHT ►        FPS  heading  status    │
        │          ▼ BACK                                              │
        └──────────────────────────────────────────────────────────────┘
        """
        DASH_H   = 160
        BG       = (18, 18, 18)
        ACTIVE   = (0, 220, 80)      # bright green  – direction available
        INACTIVE = (55, 55, 55)      # dark gray      – direction unavailable
        DIVIDER  = (60, 60, 60)

        panel = np.full((DASH_H, w, 3), BG, dtype=np.uint8)

        # ── divider line between camera and dashboard ──────────────
        cv2.line(panel, (0, 0), (w, 0), (80, 80, 80), 1)

        # ══════════════════════════════════════════════════════════
        #  TOP HALF  –  OFF-CENTER error bar   (rows 10–65)
        # ══════════════════════════════════════════════════════════
        BAR_X0, BAR_X1 = 130, w - 20
        BAR_MID        = (BAR_X0 + BAR_X1) // 2
        BAR_HALF       = (BAR_X1 - BAR_X0) // 2
        BAR_Y0, BAR_Y1 = 18, 42
        MAX_ERR        = float(w // 2)

        # Label
        cv2.putText(panel, "OFF CENTER", (8, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Background track
        cv2.rectangle(panel, (BAR_X0, BAR_Y0), (BAR_X1, BAR_Y1),
                      DIVIDER, -1)
        cv2.rectangle(panel, (BAR_X0, BAR_Y0), (BAR_X1, BAR_Y1),
                      (90, 90, 90), 1)

        # Centre tick
        cv2.line(panel, (BAR_MID, BAR_Y0 - 3), (BAR_MID, BAR_Y1 + 3),
                 (140, 140, 140), 1)

        # Filled error region (green→yellow→red based on magnitude)
        err_clamped = max(-MAX_ERR, min(MAX_ERR, lateral_error))
        bubble_x    = int(BAR_MID + (err_clamped / MAX_ERR) * BAR_HALF)
        ratio       = abs(err_clamped) / MAX_ERR           # 0..1
        bar_colour  = (
            int(0   + ratio * 0),
            int(220 - ratio * 220),
            int(80  + ratio * 175),
        )
        if err_clamped >= 0:
            cv2.rectangle(panel, (BAR_MID, BAR_Y0 + 2),
                          (bubble_x, BAR_Y1 - 2), bar_colour, -1)
        else:
            cv2.rectangle(panel, (bubble_x, BAR_Y0 + 2),
                          (BAR_MID, BAR_Y1 - 2), bar_colour, -1)

        # Bubble marker
        cv2.circle(panel, (bubble_x, (BAR_Y0 + BAR_Y1) // 2),
                   9, (255, 255, 255), -1)
        cv2.circle(panel, (bubble_x, (BAR_Y0 + BAR_Y1) // 2),
                   9, (0, 0, 0), 1)

        # Value text
        norm_error = lateral_error / MAX_ERR if MAX_ERR > 0 else 0.0
        direction_txt = "RIGHT" if lateral_error > 1 else ("LEFT" if lateral_error < -1 else "CENTER")
        val_colour    = (0, 200, 80) if abs(lateral_error) < 15 else (0, 100, 255)
        cv2.putText(panel,
                    f"{lateral_error:+.1f} px  norm {norm_error:+.3f}  {direction_txt}",
                    (BAR_X0, BAR_Y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, val_colour, 1)

        # ══════════════════════════════════════════════════════════
        #  BOTTOM HALF  –  D-pad direction arrows   (rows 75–155)
        # ══════════════════════════════════════════════════════════
        CX = w // 2        # D-pad horizontal centre
        CY = 115           # D-pad vertical centre
        AW = 22            # arrow half-width
        AH = 20            # arrow height

        def arrow_colour(available):
            return ACTIVE if available else INACTIVE

        def filled_triangle(pts, colour):
            cv2.fillPoly(panel, [np.array(pts, dtype=np.int32)], colour)

        # ── FORWARD (up) ──
        filled_triangle([(CX, CY - AH - 20),
                          (CX - AW, CY - 20),
                          (CX + AW, CY - 20)],
                         arrow_colour(directions["forward"]))
        cv2.putText(panel, "FWD",
                    (CX - 20, CY - AH - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    arrow_colour(directions["forward"]), 1)

        # ── BACK (down) ──
        filled_triangle([(CX, CY + AH + 20),
                          (CX - AW, CY + 20),
                          (CX + AW, CY + 20)],
                         arrow_colour(directions["backward"]))
        cv2.putText(panel, "BACK",
                    (CX - 24, CY + AH + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    arrow_colour(directions["backward"]), 1)

        # ── LEFT ──
        filled_triangle([(CX - AH - 60, CY),
                          (CX - 60, CY - AW),
                          (CX - 60, CY + AW)],
                         arrow_colour(directions["left"]))
        cv2.putText(panel, "LEFT",
                    (CX - AH - 100, CY + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    arrow_colour(directions["left"]), 1)

        # ── RIGHT ──
        filled_triangle([(CX + AH + 60, CY),
                          (CX + 60, CY - AW),
                          (CX + 60, CY + AW)],
                         arrow_colour(directions["right"]))
        cv2.putText(panel, "RIGHT",
                    (CX + AH + 64, CY + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    arrow_colour(directions["right"]), 1)

        # ── Centre dot of D-pad ──
        cv2.circle(panel, (CX, CY), 6, (70, 70, 70), -1)

        # ── Info column (right side) ──
        INFO_X = w - 160
        cv2.putText(panel, f"FPS  {fps:.1f}",
                    (INFO_X, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        if heading_deg is not None:
            cv2.putText(panel, f"HDG  {heading_deg:+.1f}",
                        (INFO_X, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        status_txt    = "PATH OK" if directions["forward"] else "NO PATH"
        status_colour = (0, 200, 80) if directions["forward"] else (0, 60, 220)
        cv2.putText(panel, status_txt,
                    (INFO_X, 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_colour, 2)

        return panel

    def _maybe_save(self, debug_frame: np.ndarray):
        if PATH_SAVE_DEBUG_EVERY_N > 0 and \
                self._frame_count % PATH_SAVE_DEBUG_EVERY_N == 0:
            ts    = time.strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(
                PATH_DEBUG_SAVE_DIR,
                f"dbg_{ts}_{self._frame_count:06d}.jpg")
            cv2.imwrite(fname, debug_frame)

    @staticmethod
    def _empty_result(frame: np.ndarray, mask: np.ndarray,
                      lateral_error: float) -> dict:
        return {
            "mask":          mask,
            "contour":       None,
            "center_points": [],
            "lateral_error": lateral_error,
            "heading_deg":   None,
            "debug_frame":   frame.copy(),
        }


_line_ws = None


def _get_line_ws():
    """Return a connected WebSocket, reconnecting if necessary."""
    global _line_ws
    if _line_ws is not None and _line_ws.connected:
        return _line_ws
    try:
        _line_ws = websocket.WebSocket()
        _line_ws.connect(LINE_SENSOR_WS_URL, timeout=2)
        _line_ws.settimeout(0.05)
        print(f"[WS] Connected to {LINE_SENSOR_WS_URL}")
    except Exception as e:
        print(f"[WS] Connection failed: {e}")
        _line_ws = None
    return _line_ws


def send_line_data(path_result, frame_w, frame_h):
    """Send line-sensor data over a persistent WebSocket (fire-and-forget)."""
    global _line_ws
    directions  = path_result["directions"]
    lat_err     = path_result["lateral_error"]
    heading     = path_result["heading_deg"]
    has_path    = path_result["contour"] is not None

    half_w      = frame_w / 2.0
    norm_error  = lat_err / half_w if half_w > 0 else 0.0

    payload = {
        "on_line":        has_path,
        "lateral_error":  round(lat_err, 2),
        "normalized_error": round(norm_error, 3),
        "heading_deg":    round(heading, 2) if heading is not None else None,
        "paths": {
            "forward":  directions["forward"],
            "backward": directions["backward"],
            "left":     directions["left"],
            "right":    directions["right"],
        },
        "frame_width":  frame_w,
        "frame_height": frame_h,
        "timestamp":    time.time(),
    }

    ws = _get_line_ws()
    if ws is None:
        return
    try:
        ws.send(json.dumps(payload))
    except Exception:
        _line_ws = None


def configure_camera():
    """Push settings to ESP32-CAM before the stream starts."""
    config_urls = [
         f"http://{CAMERA_IP}/control?var=framesize&val=10",
         f"http://{CAMERA_IP}/control?var=quality&val=20",
         f"http://{CAMERA_IP}/control?var=brightness&val=-2",
        f"http://{CAMERA_IP}/control?var=awb_gain&val=1",
         f"http://{CAMERA_IP}/control?var=wb_mode&val=0",
        f"http://{CAMERA_IP}/control?var=aec&val=1",
        f"http://{CAMERA_IP}/control?var=wpc&val=0",
        f"http://{CAMERA_IP}/control?var=lenc&val=0",
        f"http://{CAMERA_IP}/control?var=vflip&val=1",
        f"http://{CAMERA_IP}/control?var=special_effect&val=2",

        # f"http://{CAMERA_IP}/control?var=awb&val=0",
        # f"http://{CAMERA_IP}/control?var=agc&val=0",
        # f"http://{CAMERA_IP}/control?var=raw_gma&val=0",
        # f"http://{CAMERA_IP}/control?var=agc_gain&val=10",
        # f"http://{CAMERA_IP}/control?var=aec_value&val=429",
    ]
    print("Configuring camera …")
    for url in config_urls:
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code != 200:
                print(f"  FAILED: {url}  →  {r.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"  FAILED: {url}  →  {e}")
            return False
    print("Camera configured.")
    return True


def open_camera_stream():
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        return None
    return cap


# ──────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────────────────

def main():
    if not configure_camera():
        print("Camera configuration failed – aborting.")
        return

    cap = open_camera_stream()
    if cap is None:
        print("Cannot open stream.  Check STREAM_URL and network.")
        return

    path_detector      = PathDetector()
    prev_time          = time.time()
    last_line_send     = 0.0
    line_send_interval = LINE_SENSOR_INTERVAL_MS / 1000.0
    failed_reads       = 0

    print("Running.  Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            failed_reads += 1
            print(f"Frame read failed – retrying … ({failed_reads}/{CAMERA_READ_FAILURE_LIMIT})")

            # Keep the OpenCV window responsive while the stream is unhealthy.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if failed_reads < CAMERA_READ_FAILURE_LIMIT:
                time.sleep(0.05)
                continue

            print("Too many failed reads. Reconnecting camera stream …")
            cap.release()
            time.sleep(CAMERA_RECONNECT_DELAY_SEC)
            cap = open_camera_stream()
            failed_reads = 0

            if cap is None:
                print("Reconnect failed. Will keep trying …")
                time.sleep(CAMERA_RECONNECT_DELAY_SEC)
            else:
                print("Camera stream reconnected.")
            continue

        failed_reads = 0

        # Correct camera orientation (180° flip)
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        canvas_h, canvas_w = frame.shape[:2]

        # ── FPS calculation ────────────────────────────────────────
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now

        # ── Path detection ─────────────────────────────────────────
        path_result = path_detector.detect_path(frame)

        # ── Send line sensor data every LINE_SENSOR_INTERVAL_MS ────
        now = time.time()
        if now - last_line_send >= line_send_interval:
            send_line_data(path_result, canvas_w, canvas_h)
            last_line_send = now

        # Rebuild the dashboard with real FPS (detect_path uses 0.0 as placeholder).
        cam_h    = frame.shape[0]
        display  = path_result["debug_frame"].copy()
        new_dash = PathDetector._draw_dashboard(
            frame.shape[1],
            path_result["lateral_error"],
            path_result["heading_deg"],
            path_result["directions"],
            fps)
        display[cam_h:, :] = new_dash

        # ── Display ────────────────────────────────────────────────
        cv2.imshow("Robot Navigation", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
