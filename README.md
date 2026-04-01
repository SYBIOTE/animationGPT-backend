# AnimationGPT

This project’s backend is derived from [MotionGPT](https://github.com/OpenMotionLab/MotionGPT).

## Setup

### Base dependencies

Besides cloning the code from GitHub, to run the model you also need to download and install the following.

1. Install Python dependencies: `pip install -r requirements.txt`
2. Download the model assets the project depends on
3. Download the **HumanML3D** dataset  
   Note: the upstream project uses the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset, but it only ships **`KIT_ML`**; that data must be prepared manually.

The original project provides scripts to fetch dependencies; you can also download files manually from the linked sites.

```bash
bash prepare/download_smpl_model.sh
bash prepare/prepare_t5.sh
bash prepare/download_t2m_evaluators.sh

bash prepare/download_pretrained_models.sh
```

- Dependency bundle: [Google Drive](https://drive.google.com/drive/folders/10s5HXSFqd6UTOkW2OMNc27KGmMLkVc2L)
- Model weights: [Hugging Face](https://huggingface.co/OpenMotionLab)

### Translation API

This project accepts **Chinese and English** input, but the LLM only accepts **English**, so Chinese prompts are sent to an external translation API.

Supported providers are listed below. To enable translation, copy [`configs/translate.example.json`](./configs/translate.example.json), remove the `example` suffix, and fill in the keys for the provider you use.

| Name | `kind` | Credentials |
|------|--------|-------------|
| [Youdao AI Open Platform](https://ai.youdao.com/DOCSIRMA/html/trans/api/wbfy/index.html) | `youdao` | `appKey`, `appSecret` |

Note: only **one** translation backend can be active at runtime.

## OOP wrapper

To make later development easier, model usage is wrapped in an OOP style as the [`T2MBot`](./server/bot.py) object. On construction it:

1. Loads config and ensures output directories exist  
2. Sets the `torch` seed and picks the compute device  
3. Loads `data_module` and `state_dict` to build the model  

After that, call `generate_motion` to generate motion from text.

## Caching

To avoid regenerating the same prompts (especially for bundled examples), the server caches results: if a prompt was already generated, the stored output is returned to save compute.

Behavior:

1. On startup, load previously generated result IDs from the cache directory into a set  
2. For each new request, hash the prompt to an ID  
3. If the ID is in the set → return the cached result; otherwise generate, then add the ID to the set  
4. When the set exceeds the configured maximum, evict entries  
   The implementation randomly removes a fraction of the cap to balance speed and consistency: it updates the in-memory set first, then deletes files on disk in a background thread  

Observed on-disk size per generation is on the order of **~500 KB**; with a cap of **2000** generations, expect roughly **~1 GB** of disk usage.

## JSON API (SaaS / HY-Motion–compatible)

A **slim API-only extract** is maintained as the **`animationgpt-api`** repository (git submodule at repo root: same `mGPT` + configs + `POST /v1/motion`; no Gradio, BVH, or translation stack).

For integration with **nirvana-animate-saas** (multi-provider motion), use either:

### FastAPI (recommended, same style as HY-Motion)

From the **animationGPT-backend** repo root:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8082
```

- **`GET /health`** — `{ "status": "ok" }`
- **`POST /v1/motion`** — body and response match HY-Motion’s JSON shape; `meta` includes `provider: "animationgpt"`.
- Optional env: **`MOTION_FPS`** (default **20**), **`T2M_CACHE_FOLDER`**.

### Flask (`server/main.py`)

Same JSON surface as FastAPI: **`GET /health`** and **`POST /v1/motion`** on port 8082 (no MP4/BVH download routes).

---

**Response shape:** `{ "motion": { rot6d, transl, keypoints3d, root_rotations_mat, num_frames, fps }, "meta": { text, duration, seed, provider } }` — same `motion` as HY-Motion; `fps` is typically **20** for HumanML3D.

Set **`ANIMGPT_API_URL`** in the Next.js app to this service’s base URL (e.g. `http://localhost:8082`).
