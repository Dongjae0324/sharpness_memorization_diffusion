# Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes [ICML 2025]

**Official PyTorch implementation** of
*Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes*.

If you have any questions about this work, please contact Dongjae (dongjae0324@yonsei.ac.kr)

---

## Summary

This repository provides:

- **Memorization detection** in Stable Diffusion (versions 1 & 2)
- **Inference-time memorization mitigation** (SAIL: SharpnessAware Initialization for Latent Diffusion)

You can find an example Jupyter notebook for SAIL in the [`examples/`](./examples) directory.

> **Note:**
> This repository does **not** include the detailed implementation of *Arnoldi iteration* for eigenvalue analysis.
> For reference, see [`./utils/arnoldi_iteration_jvp`](./utils/arnoldi_iteration_jvp) and the [Arnoldi iteration Wikipedia page](https://en.wikipedia.org/wiki/Arnoldi_iteration).

---

## Preparation

### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate sail
```

---

## Memorization Detection

> The following example uses **Stable Diffusion v1.4**.
> The same logic applies to **v2.0**.

### 1. Run detection

- `sd1_mem.txt` and `sd1_nmem.txt` are prompt sets for **memorized** and **non-memorized** prompts.
- You can directly run detection experiments using [`run_detection.sh`](./run_detection.sh).
- `gen_num` refers to **n**, and `hvp_sampling_num` refers to **steps** in Table 1 of our paper.

#### (a) Memorized prompts

```bash
python detect_mem.py --sd_ver 1 --data_path "prompts/sd1_mem.txt" --gen_num 4
```

#### (b) Non-memorized prompts

```bash
python detect_mem.py --sd_ver 1 --data_path "prompts/sd1_nmem.txt" --gen_num 4
```

---

### 2. Evaluate detection results

After running detection, the following files will be generated:

- `./det_outputs/sd1_mem_gen4.pt`
- `./det_outputs/sd1_nmem_gen4.pt`

You can then evaluate **AUC**, **TPR@1%FPR**, and **TPR@3%FPR**:

```bash
python detect_eval.py
```

---

## Memorization Mitigation (SAIL)

### 1. Prepare training images for memorized prompts

- Manually extract images from `prompts/sd1_mem.jsonl`.
- In our experiment, 454/500 training images were extracted and used for Stable Diffusion v1.4.
- See Appendix D of the paper for details.
- **Note:** The availability of training images may have changed (as of June 2025).
- We provide sample prompts in [`prompts/sample_mitigation.txt`](./prompts/sample_mitigation.txt) for testing.

---

### 2. Run inference-time mitigation

- Hyperparameters are listed in Appendix D.
- You can run SAIL using [`run_mitigation.sh`](./run_mitigation.sh):

```bash
python mitigate_mem.py --sd_ver 1 --data_path "prompts/sample_mitigation.txt" --gen_num 4
```

---

### 3. Integrate with SSCD & CLIP evaluation

After generating images:

- Use the training images prepared in step 1.
- Build your evaluation tool based on [`mitigate_eval.py`](./mitigate_eval.py).
- Example code for **SSCD** and **CLIP** evaluation is provided there.

For **Stable Diffusion v2.0**:

- Use multiple training images containing keywords such as `"design"`, `"designart"`, `"mills"`, and `"anderson"`.
- Take the **max** SSCD similarity score across these images, as they may have multiple similar training samples.

---

## Acknowledgements

This repository includes utilities from:

- [Yuxin Wen&#39;s diffusion_memorization](https://github.com/YuxinWenRick/diffusion_memorization)
- [Layer6&#39;s dgm_geometry](https://github.com/layer6ai-labs/dgm_geometry)

Memorized prompts are largely from:

- [Ryan Webster&#39;s onestep-extraction](https://github.com/ryanwebster90/onestep-extraction)

Non-memorized prompts are sourced from:

- **COCO**
- **Tuxemon**
- **GPT-4**
- **Lexica**

GPT-generated prompts are from:

- [Jie’s MemAttn](https://github.com/renjie3/MemAttn)

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
  jeon2025understanding,
  title={Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes},
  author={Dongjae Jeon and Dueun Kim and Albert No},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
}
```
