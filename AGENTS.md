# Working on Humaneness Podcast Server

Read README.md, docs/PODCAST_SCRIPT_WRITING_GUIDE.md, docs/API.md and docs/OPERATIONS.md first. The exact writer instruction is docs/SCRIPT_WRITER_SYSTEM_PROMPT.txt. docs/TESTING.md separates real inference from test fixtures and lists known quality limits.

Preserve the approved recipe: full Podcast checkpoint, sparse spontaneous cues, temperature 1.0, CFG 2.5, default N=4 and the documented 50/50 genuineness/reference-cosine reward. Do not silently add WER eligibility filtering, voice conversion, DPO, or a different checkpoint. N is independent of GPU microbatch size.

Public script input must never select arbitrary server paths, commands, Python code or download URLs. Keep job capabilities, scripts, generated audio, source reference recordings, model weights and machine-specific assets/deployment files outside Git. Run `python scripts/check_release.py` before publication.

The queue has one GPU worker. Do not kill other services to free VRAM. The explicit recovery integration test kills only the newly created test job's owned subprocess. Use a new job after modifying runtime/provenance; in-flight saved candidates are not silently mixed with changed code or references.

Run `python -m pytest -q tests` after changes to the contract, queue or assembly. Runtime changes additionally need a real-model job and a truthful report of timings/errors. A synthetic signal test is not evidence of speech quality. Revalidate the public reverse-proxy route and existing host applications after integration changes.

Vendored source is pinned and hashed. If deliberately changing it, record upstream attribution, the exact modification and new checksum rather than pretending it is unchanged upstream code. Use the model links/revisions in README and preparation code for reproducible setup.
