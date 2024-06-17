import torch


def get_optimizer(optim_name, model, args):
    optim_name = optim_name.lower()
    model_weights = filter(lambda p: p.requires_grad, model.parameters())

    if optim_name == 'adam':
        optimizer = torch.optim.Adam(model_weights, lr=args.learning_rate, weight_decay=args.reg_weight)
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model_weights, lr=args.learning_rate, weight_decay=args.reg_weight)
    elif optim_name == 'sgd':
        optimizer = torch.optim.SGD(model_weights, lr=args.learning_rate, momentum=args.momentum,
                                    nesterov=True, weight_decay=args.reg_weight)
    elif optim_name == 'lion':
        from lion_pytorch import Lion
        optimizer = Lion(model_weights, lr=args.learning_rate, weight_decay=args.reg_weight)
    elif optim_name == 'adabound':
        import adabound
        optimizer = adabound.AdaBound(model_weights, lr=args.learning_rate, final_lr=0.1)
    elif optim_name == 'prodigy':
        from prodigyopt import Prodigy
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy(model_weights, lr=1., weight_decay=args.reg_weight)
    elif optim_name == 'd_adam':
        from dadaptation import DAdaptAdam
        optimizer = DAdaptAdam(model_weights, lr=1., )
    elif optim_name == 'd_lion':
        from dadaptation import DAdaptLion
        optimizer = DAdaptLion(model_weights, lr=1., )
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(optim_name))

    return optimizer


def get_lr_scheduler(optimizer, total_steps, args):
    if args.lr_schedule == 'warmup_cosine':
        from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_steps,
                                                  max_epochs=total_steps)
    elif args.lr_schedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        # if args.checkpoint is not None:
        #     scheduler.step(epoch=start_epoch)
    elif args.lr_schedule == 'poly':
        def lambdas(epoch):
            return (1 - float(epoch) / float(total_steps)) ** 0.9

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    else:
        raise NotImplementedError
    return scheduler
