# AnimationGPT

This project's backend is built upon [MotionGPT](https://github.com/OpenMotionLab/MotionGPT).

## Setup

### Dependencies

In addition to cloning the code from GitHub, the following dependency files need to be downloaded and installed to use this model properly.

1. Install dependencies via `pip install -r requirement`
2. Download the required model files
3. Download the `HumanML3D` dataset
    > Note: the original project uses the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset, but it only provides the `KIT_ML` dataset, which requires manual processing.

The original project also provides scripts for downloading the dependencies. You can also download them manually from the linked sites.

```bash
bash prepare/download_smpl_model.sh
bash prepare/prepare_t5.sh
bash prepare/download_t2m_evaluators.sh

bash prepare/download_pretrained_models.sh
```

-   Dependency files: [Google Drive](https://drive.google.com/drive/folders/10s5HXSFqd6UTOkW2OMNc27KGmMLkVc2L)
-   Model files: [Huggingface](https://huggingface.co/OpenMotionLab)

### Translation API

This project supports both Chinese and English input, but the LLM only accepts English. An external API is used to translate Chinese input into English.
The currently supported APIs are listed below.
To use the translation service, apply for an API key from the corresponding provider,
then copy [configs/translate.example.json](./configs/translate.example.json)
and remove the `example` suffix. Fill in the required keys accordingly.

| Name | Kind | Required Keys |
|--|--|--|
| [Youdao AI](https://ai.youdao.com/DOCSIRMA/html/trans/api/wbfy/index.html) | youdao | `appKey`, `appSecret` |

> Note: only one translation service can be active at runtime.

## OOP Architecture

To facilitate future development, this project uses OOP (Object-Oriented Programming) to encapsulate model operations into the [`T2MBot`](./server/bot.py) object. When created, this object goes through the following initialization steps:

1. Load configuration files and create output directories
2. Set the `torch` seed and select the compute device
3. Sequentially load `data_module` and `state_dict` to build the model

After initialization, users can call `generate_motion` to generate motion from text.

## Caching

To avoid redundant generation for identical prompts — especially for the example prompts we provide — a caching mechanism is used. When a previously generated prompt is requested again, the server returns the cached result directly, reducing computational overhead.

The caching strategy works as follows:

1. On startup, the server reads previously generated result IDs from the cache directory into an in-memory set
2. For each new generation request, the prompt is hashed to produce an ID
3. The ID is checked against the set:
    - If it exists, the cached result is returned immediately
    - If not, the motion is generated and the ID is added to the set
4. When the number of cached results exceeds a configured maximum, cache eviction is triggered
    > The server randomly deletes n% of the maximum number of cached results. For efficiency and data consistency, it first removes entries from the in-memory set, then deletes the corresponding files from disk using multiple threads.

Based on observation, the total size of server-generated files is roughly 500 KB per result. Setting the maximum cache size to 2000 entries would consume approximately 1 GB of disk space.
