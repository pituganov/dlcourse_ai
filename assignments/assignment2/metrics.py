def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    # predicted_labels = prediction.argmax(axis=1)
    accuracy = (prediction == ground_truth).sum()
    accuracy /= prediction.shape[0]

    return accuracy
