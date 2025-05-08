from pytorch_fid import fid_score


# 计算两个文件夹图像的FID
path1 = r"./DIR-D/testing/gt"
path2 = r"./result/Final_Image"


fid_value = fid_score.calculate_fid_given_paths(
    paths=[path1, path2],
    batch_size=1,
    device='cuda:0',
    dims=2048
)
print(f'FID: {fid_value:.2f}')
