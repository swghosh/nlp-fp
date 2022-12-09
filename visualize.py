from matplotlib import pyplot as plt
import glob
import json

def get_epoch_losses(path):
    print(path)
    checkpoint_paths = glob.glob(path + '/checkpoint*/trainer_state.json')
    sorted(checkpoint_paths)

    ckpt_path_to_use = checkpoint_paths[-1]

    epochs = []
    losses = []
    with open(ckpt_path_to_use) as f:
        obj = json.load(f)
        log_history = obj['log_history']
        for l in log_history:
            epochs.append(l['epoch'])
            losses.append(l['loss'])
    return epochs, losses


model_paths = {
    'AdamW with Constant LR': '/tmp/model-constant-lr',
    'AdamW with Cosine Restarts LR Schedule': '/tmp/model-cosine-restarts',
    'Adagrad with Warmup and Linear LR Schedule': '/tmp/model-adagrad',
    'SGD with Cosine Restarts LR Schedule (a.k.a. SGDR)': '/tmp/model-sgdr'
}

plt.style.use('ggplot')
for key in model_paths:
    ckpt_path = model_paths[key]
    epochs, losses = get_epoch_losses(ckpt_path)
    plt.plot(epochs, losses, marker='.', linestyle='--', label=key)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Fine-tuning ElectraSmall on SQuAD\nfor different optimizers and LR schedules')
plt.savefig('figure.pdf')
