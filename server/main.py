from flask import Flask, jsonify, request
import logging

from bot import T2MBot
from motion_json import DEFAULT_FPS, joints_to_motion_data

app = Flask("animationGPT")
with app.app_context():
    bot = T2MBot()

    logging.basicConfig(
        filename='results/animation.log',
        format="%(asctime)s: [%(levelname)s] %(message)s ",
        level=logging.INFO
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/v1/motion", methods=["POST"])
def v1_motion():
    """
    JSON API compatible with nirvana-animate-saas MotionGenerateResponse.
    Body: { "text": string, "duration"?: number, "seed"?: number, "cfg_scale"?: number }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if text is None or not str(text).strip():
        return jsonify({"error": 'Invalid body: "text" (non-empty string) is required.'}), 400
    text = str(text).strip()

    duration = data.get("duration", 3.0)
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 3.0

    seed = data.get("seed", 42)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 42

    try:
        joints, lengths, _feats = bot.infer_motion_tensors(text)
        if lengths <= 0:
            return jsonify({"error": "Model returned empty motion."}), 500
        j = joints[:lengths].detach().cpu().numpy()
        motion = joints_to_motion_data(j, lengths, fps=DEFAULT_FPS)
        return jsonify(
            {
                "motion": motion,
                "meta": {
                    "text": text,
                    "duration": duration,
                    "seed": seed,
                    "provider": "animationgpt",
                },
            }
        )
    except Exception as e:
        logging.exception("v1/motion failed: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # from waitress import serve
    # serve(app, port=8082, host="0.0.0.0", threaded=True)
    app.run(port=8082, host="0.0.0.0", threaded=True)
