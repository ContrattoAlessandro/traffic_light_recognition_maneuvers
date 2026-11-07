"""Junction relevance for DTLD: which lamps belong to the junction the ego is approaching.

DTLD's ``relevant`` flag marks the lamps that serve the ego's planned route. The junction label adds
the other front-facing vehicle lamps of the same junction (other lanes' heads), and leaves out lamps of
farther junctions. A non-route lamp belongs to the junction when the median of its height over its
whole track (DTLD ``track_id`` within a session), measured relative to the median route-lamp height of
each frame, is at least RATIO. Lamps of the same physical size at the same distance have a ratio near
1; a lamp of the next junction is farther away and its ratio is lower (and falls as the ego approaches).
Using the track median rather than each frame's ratio keeps a lamp's label stable over a sequence.

Frames without a route lamp give no reference: their lamps are not junction lamps.
Only vehicle lamps (circle, arrows, unknown pictogram) facing the camera can be junction lamps.
"""
import collections
import statistics

VEHICLE = {"circle", "arrow_left", "arrow_right", "arrow_straight", "arrow_straight_left", "unknown"}
OTHER = {"pedestrian", "bicycle", "pedestrian_bicycle", "tram"}
RATIO = 0.85


def session(im):
    return "/".join(im["image_path"].replace("\\", "/").split("/")[:-1])


def front_vehicle(im):
    return [l for l in im["labels"] if l["attributes"]["direction"] == "front" and l["attributes"]["pictogram"] in VEHICLE]


def track_ratios(images):
    """(session, track_id) -> median height ratio to the frame's route lamps, over the frames that have route lamps."""
    obs = collections.defaultdict(list)
    for im in images:
        front = front_vehicle(im)
        route = [l for l in front if l["attributes"]["relevance"] == "relevant"]
        if not route:
            continue
        h = statistics.median(l["h"] for l in route)
        for l in front:
            if l["attributes"]["relevance"] != "relevant":
                obs[(session(im), l["track_id"])].append(l["h"] / h)
    return {k: statistics.median(v) for k, v in obs.items()}


def junction_role(im, ratios, ratio=RATIO):
    """Per label of ``im`` (same order): "route", "junction" (other lamp of the same junction) or None."""
    front = front_vehicle(im)
    has_route = any(l["attributes"]["relevance"] == "relevant" for l in front)
    out = []
    for l in im["labels"]:
        if l not in front or not has_route:
            out.append(None)
        elif l["attributes"]["relevance"] == "relevant":
            out.append("route")
        else:
            out.append("junction" if ratios.get((session(im), l["track_id"]), 0) >= ratio else None)
    return out
