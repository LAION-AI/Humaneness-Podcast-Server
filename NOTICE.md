# Attribution and licenses

New server code, documentation and script examples: Copyright 2026 LAION and Humaneness Podcast contributors, **CC BY 4.0**. Attribute “Humaneness Podcast Server, LAION” and link to this repository and the license; indicate modifications.

The vendored inference/scoring implementation was extracted from [LAION-AI/Humaneness-Voice](https://github.com/LAION-AI/Humaneness-Voice), commit `65d01782883df466d943d7e6e31006cc68017f61`, `podcast_pipeline/`. File hashes are in `vendor/podcast_pipeline/SOURCE.json`. Original notices are retained under `vendor/podcast_pipeline/runtime/`. The new service wraps these functions with a durable API, independent job checkpoints, bounded scheduling, explicit pause assembly and 128 kbit/s MP3 export; it does not retrain or modify the model weights.

Model code downloaded from the pinned Podcast and MOSS-v2 repositories, PyTorch/Transformers/SpeechBrain, the MOSS audio codec, ECAPA, Humaneness Ears, Parakeet and the Qwen aligner retain their own published licenses and attributions. See each model repository linked in README. The older MMS alignment code is present in the attributed upstream alignment module, but this service uses Qwen and Parakeet; it does not download or run the legacy noncommercial MMS aligner.

Example reference MP3s and acoustic codes are downloaded separately from the pinned Podcast repository. Their provenance is retained at `vendor/podcast_pipeline/references/PROVENANCE.json`. The model/server license does not relicense source recordings or grant rights to another person's identity. This repository contains no model weight files, reference recordings, user audio, private job data or API credentials.

The exact prompt recipe and its listener-preference history are linked in the writing guide and upstream documentation. Some copied upstream documentation contains relative links intended for its original repository; use README's model/manual links for those original destinations.
