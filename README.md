vlms
- NanoVLM intro: https://huggingface.co/blog/nanovlm
- Nice VLMs survey: https://huggingface.co/blog/vlms-2025

Videos
- Qwen: https://www.youtube.com/watch?v=b0xlsQ_6wUQ
    Talks about hybrid thinking mode. Would be interesting to try VL models while building my VLM.

Dev Env
- To activate the uv env, use source .venv/bin/activate
- To run the train script, use `python -u train.py > /tmp/logs.txt 2>&1`

Notes
- Dataloader design can be simplified by returning the full input sequence from the datasets.py


Experimental Results
- 1 GPU training can get 50% accuracy on mmstar (best accuracy)

Next steps
- Evals
- Incorporate other changes from the main repo
- Add more datasets including image captioning.
