def transform(x, mean, std):
        return (x - mean) / std

def reverse_transform(x, mean, std):
    return (x * std) + mean