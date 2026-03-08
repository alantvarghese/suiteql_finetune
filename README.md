8) Runpod setup steps

Runpod’s docs say the easiest path is to launch a Pod with an official PyTorch/Jupyter-compatible template, then connect through the Pod’s Connect flow; SSH is also supported on official templates.

On Runpod

Launch a Runpod PyTorch pod.

Pick a GPU like RTX 4090, A5000/A6000, or A100.

Connect via JupyterLab or SSH.

In the pod
cd /workspace
git clone <your-repo-url> netsuite-finetune
cd netsuite-finetune

python -m pip install --upgrade pip
pip install -r requirements.txt

If your base model is gated on Hugging Face:

huggingface-cli login

Then place your dataset files in /workspace/data/ and run:

bash run_train.sh
9) What to do after training

Your LoRA adapter will be in:

/workspace/outputs/netsuite-sql-lora

Test it:

python infer.py
10) Good first hyperparameters for your project

For a first pass on NetSuite SQL generation, I’d use:

r=16

lora_alpha=16

max_seq_length=2048

learning_rate=2e-4

epochs=3

batch size effectively 2 x 8 accumulation = 16

Then evaluate on:

query syntax validity

table selection accuracy

join correctness

whether it reduces retrieval/context compared to your RAG baseline

That evaluation framing matches what you told me about the project goal.

11) Important caution

Do not train it only on “question → SQL” pairs.

Also include examples for:

wrong query → corrected query

question → table selection

question → join path

question → SQL + short explanation

negative examples where the model should say it lacks enough schema context

That will make the model far more useful.
