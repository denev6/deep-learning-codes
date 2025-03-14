from torchinfo import summary


def print_model(model, input_size=None, *args, **kwargs):
    """Print model summary.

    Args:
        model: model to summarize.
        input_size: input size for model include batch.
    """
    summary(model, input_size, *args, **kwargs)
