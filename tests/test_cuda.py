def test_cuda():
    import torch
    print(torch.cuda.is_available())