import requests

def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"文件已成功下载到 {save_path}")
    else:
        print(f"下载失败，错误代码：{response.status_code}")

# 示例：GitHub 上的原始文件 URL 和保存路径
url = "https://raw.githubusercontent.com/myleott/mnist_png/blob/master/mnist_png.tar.gz"
save_path = "mnist_png.tar.gz"

# 下载文件
download_file(url, save_path)
