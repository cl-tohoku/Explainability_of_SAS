from captum.attr import Saliency, IntegratedGradients, InputXGradient
from typing import Any, Callable

import torch
from captum._utils.common import _format_input, _format_output, _is_tuple
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage


from captum._utils.common import _run_forward, _format_additional_forward_args


def compute_gradients(
    forward_fn,
    inputs,
    target_ind,
    knn_model,
    additional_forward_args: Any = None,
):
    r"""
    Computes gradients of the output with respect to inputs for an
    arbitrary forward function.
    Args:
        forward_fn: forward function. This can be for example model's
                    forward function.
        input:      Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        additional_forward_args: Additional input arguments that forward
                    function requires. It takes an empty tuple (no additional
                    arguments) if no additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(
            forward_fn, inputs, target_ind, additional_forward_args)
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )

        inputs = _format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args)

        logits = forward_fn(
            *(*inputs, *additional_forward_args)
            if additional_forward_args is not None
            else inputs
        )

        cpu_logits = logits.detach().to("cpu").numpy()
        # print(cpu_logits.shape)
        # print(target_ind)
        positive_embeddings = knn_model.get_same_label_embeddings(
            cpu_logits, label=target_ind, k=1)
        # print(positive_embeddings.shape)
        negative_embeddings = knn_model.get_different_label_embeddings(
            cpu_logits, label=target_ind, k=1)
        positive_embeddings = torch.FloatTensor(
            positive_embeddings).to("cuda")  # .view(1, -1)
        negative_embeddings = torch.FloatTensor(
            negative_embeddings).to("cuda")  # .view(1, -1)

        loss = triplet_loss(logits, positive_embeddings,
                            negative_embeddings)

        grads = torch.autograd.grad(torch.unbind(loss), inputs)
    return grads


def euclid_dist(x, y):
    return (x - y).pow(2).sum(1).pow(0.5)


def triplet_loss(target, positive, negative):
    # return torch.nn.functional.relu(euclid_dist(target, positive) - euclid_dist(target, negative) + margin)
    return euclid_dist(target, positive) - euclid_dist(target, negative)


def regularization(target_embeddings):
    _target = torch.clone(target_embeddings)
    return _target / torch.sum(_target)


class MetricSaliency(Saliency):
    def __init__(self, net_patams, knn_model):
        super().__init__(net_patams)
        self.knn_model = knn_model

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        abs: bool = True,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        # No need to format additional_forward_args here.
        # They are being formated in the `_run_forward` function in `common.py`

        # gradients = self.gradient_func(
        #     self.forward_func, inputs, target, additional_forward_args
        # )
        gradients = compute_gradients(
            self.forward_func, inputs, target, self.knn_model, additional_forward_args
        )
        if abs:
            attributions = tuple(torch.abs(gradient) for gradient in gradients)
        else:
            attributions = gradients
        undo_gradient_requirements(inputs, gradient_mask)
        return _format_output(is_inputs_tuple, attributions)


class MetricInputXGradient(InputXGradient):
    def __init__(self, net_patams, knn_model):
        super().__init__(net_patams)
        self.knn_model = knn_model

    @log_usage()
    def attribute(
        self,
        inputs,
        target=None,
        additional_forward_args=None,
    ):
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        gradients = compute_gradients(
            self.forward_func, inputs, target, self.knn_model, additional_forward_args
        )

        attributions = tuple(
            input * gradient for input, gradient in zip(inputs, gradients)
        )

        undo_gradient_requirements(inputs, gradient_mask)
        return _format_output(is_inputs_tuple, attributions)

    @property
    def multiplies_by_inputs(self):
        return True


from captum.attr._utils.approximation_methods import approximation_parameters
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)


class MetricIntegratedGradients(IntegratedGradients):
    def __init__(self, net_patams, knn_model):
        super().__init__(net_patams)
        self.knn_model = knn_model

    def _attribute(
        self,
        inputs,
        baselines,
        target=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
        step_sizes_and_alphas=None,
    ):
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        # grads = self.gradient_func(
        #     forward_fn=self.forward_func,
        #     inputs=scaled_features_tpl,
        #     target_ind=expanded_target,
        #     additional_forward_args=input_additional_args,
        # )

        grads = compute_gradients(
            self.forward_func, scaled_features_tpl, expanded_target, self.knn_model, input_additional_args
        )

        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        if not self.multiplies_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(total_grads, inputs, baselines)
            )
        return attributions
