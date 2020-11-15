import torch
import shutil
import math
import torch.optim as optim
from apex.parallel import LARC


def learning_rate_schedule(arguments):
        """Build learning rate schedule."""
        optimizer_params = arguments['optimizer'].state[arguments['optimizer'].param_groups[0]["params"][-1]]
        if 'step' in optimizer_params:
                global_step = optimizer_params['step'] // arguments['num_fixations']
        else:
                global_step = 1 // arguments['num_fixations']

        # warmup_steps = warmup_epochs * num_examples / batch_size
        warmup_steps = int(round(arguments['warmup_epochs'] * arguments['num_examples'] // arguments['batch_size']))

        global_batch_size = arguments['world_size'] * arguments['batch_size']
        if arguments['learning_rate_scaling'] == 'linear':
                # scaled_lr = base_learning_rate * global_batch_size / 256
                scaled_lr = arguments['base_learning_rate'] * global_batch_size / 256.
        elif arguments['learning_rate_scaling'] == 'sqrt':
                # scaled_lr = base_learning_rate * sqrt(global_batch_size)
                scaled_lr = arguments['base_learning_rate'] * math.sqrt(global_batch_size)
        else:
                raise ValueError('Unknown learning rate scaling {}'.format(arguments['learning_rate_scaling']))

        learning_rate = (float(global_step) / int(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

        # Cosine decay learning rate schedule
        total_steps = _get_train_steps(arguments['num_examples'], arguments['train_epochs'], arguments['batch_size'])
        learning_rate = (learning_rate if global_step < warmup_steps else _cosine_decay(scaled_lr,
                                                                                        global_step - warmup_steps,
                                                                                        total_steps - warmup_steps))

        for param_group in arguments['optimizer'].param_groups:
                param_group['lr'] = learning_rate










def _cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
        global_step = min(global_step, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return learning_rate * decayed



def _get_train_steps(num_examples, train_epochs, train_batch_size):
        """Determine the number of training steps."""
        return num_examples * train_epochs // train_batch_size + 1







def get_optimizer(model, args):
        """Returns an optimizer."""
        if args.optimizer == 'sgd':
                optimizer = optim.SGD(model.parameters(),
                                      args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay)

        elif args.optimizer == 'adam':
                optimizer = optim.Adam(model.parameters(),
                                       args.lr)

        elif args.optimizer == 'lars':
                optimizer = optim.Adam(model.parameters(),
                                       args.lr)
                optimizer = LARC.LARC(optimizer)

        else:
                raise ValueError('Unknown optimizer {}'.format(args.optimizer))

        return optimizer






def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
                print('Saving a new best model with precesion {}'.format(state['best_prec1']))
                shutil.copyfile(filename, 'model_best.pth.tar')




def top_k_accuracy(preds, target, k):
        a=torch.transpose(torch.topk(preds,k=k,dim=1)[1],0,1)
        b=torch.argmax(target,dim=1)
        c=a==b
        d=torch.any(c,dim=0)
        return torch.sum(d)/(d.shape[0] + 0.0)
