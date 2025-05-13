def test_cuda():
    import torch
    print(torch.cuda.is_available())


if __name__ == '__main__':
    test_cuda()