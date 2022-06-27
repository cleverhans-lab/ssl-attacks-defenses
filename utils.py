import os
import shutil
import torch


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'),
                  'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_victim(epochs, dataset, model, arch, loss, device, discard_mlp=False,
                watermark="False", entropy="False"):
    if watermark == "True":
        checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLR/{epochs}{arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}WATERMARK.pth.tar",
            map_location=device)
    elif entropy == "True":
        checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLR/{epochs}{arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}ENTROPY.pth.tar",
            map_location=device)
    else:
        checkpoint = torch.load(
            f"/checkpoint/{os.getenv('USER')}/SimCLR/{epochs}{arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}.pth.tar",
            map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = state_dict.copy()
    if discard_mlp:  # no longer necessary as the model architecture has no backbone.fc layers
        for k in list(state_dict.keys()):
            if k.startswith('backbone.fc'):
                del new_state_dict[k]
        model.load_state_dict(new_state_dict, strict=False)
        return model
    model.load_state_dict(state_dict, strict=False)
    return model


def load_watermark(epochs, dataset, model, arch, loss, device):
    checkpoint = torch.load(
        f"/checkpoint/{os.getenv('USER')}/SimCLR/{epochs}{arch}{loss}TRAIN/{dataset}_checkpoint_{epochs}_{loss}WATERMARK.pth.tar",
        map_location=device)
    try:
        state_dict = checkpoint['watermark_state_dict']
    except:
        state_dict = checkpoint['mlp_state_dict']

    model.load_state_dict(state_dict)
    return model


def print_args(args, get_str=False):
    if "delimiter" in args:
        delimiter = args.delimiter
    elif "sep" in args:
        delimiter = args.sep
    else:
        delimiter = ";"
    print("###################################################################")
    print("args: ")
    keys = sorted(
        [
            a
            for a in dir(args)
            if not (
                a.startswith("__")
                or a.startswith("_")
                or a == "sep"
                or a == "delimiter"
        )
        ]
    )
    values = [getattr(args, key) for key in keys]
    if get_str:
        keys_str = delimiter.join([str(a) for a in keys])
        values_str = delimiter.join([str(a) for a in values])
        print(keys_str)
        print(values_str)
        return keys_str, values_str
    else:
        for key, value in zip(keys, values):
            print(key, ": ", value, flush=True)
    print("ARGS FINISHED", flush=True)
    print("######################################################")
