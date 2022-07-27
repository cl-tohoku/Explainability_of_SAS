import torch


def get_optimizer(optim, model, ):
    if optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, alpha=0.9, eps=1e-6)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.0, weight_decay=0.0)
    elif optim == "adagrad":
        optimizer = torch.optim.Adagrad(
            filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, epsilon=1e-06)
    elif optim == "adadelta":
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=1.0, rho=0.95, epsilon=1e-06)
    elif optim == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.999),
                                     eps=1e-08)
    elif optim == "adam-bert":
        optimizer = torch.optim.Adam(
            params=[
                {"params": filter(lambda p: p.requires_grad, model.attention.parameters(
                )), "lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08},
                {"params": filter(lambda p: p.requires_grad, model.fc.parameters(
                )), "lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08},
                {"params": filter(lambda p: p.requires_grad, model.bert.parameters(
                )), "lr": 0.00001, "betas": (0.9, 0.999), "eps": 1e-08},
            ]
        )
    elif optim == "rmsprop-bert":
        optimizer = torch.optim.Adam(
            params=[
                {"params": filter(lambda p: p.requires_grad, model.attention.parameters(
                )), "lr": 0.001, "alpha": 0.9, "eps": 1e-6},
                {"params": filter(lambda p: p.requires_grad, model.fc.parameters(
                )), "lr": 0.001, "alpha": 0.9, "eps": 1e-6},
                {"params": filter(lambda p: p.requires_grad, model.bert.parameters(
                )), "lr": 0.00001, "alpha": 0.9, "eps": 1e-6},
            ]
        )
    else:
        raise ValueError(optim)

    return optimizer
