import torch
import torch.nn as nn

def kmeans(X, k, max_iters=100):
    """
    Triển khai thuật toán K-means sử dụng PyTorch.

    Tham số:
    - X: Dữ liệu đầu vào, dạng tensor PyTorch (n_samples, n_features)
    - k: Số cụm
    - max_iters: Số lần lặp tối đa

    Trả về:
    - centroids: Tọa độ các centroid cuối cùng
    - labels: Nhãn cụm cho mỗi điểm dữ liệu
    """
    indices = torch.randperm(X.size(0))[:k]
    centroids = X[indices]

    for i in range(max_iters):
        distances = torch.cdist(X, centroids, p=2)
        labels = torch.argmin(distances, dim=1)
        centroids_old = centroids.clone()
        for idx in range(k):
            if (labels == idx).sum() == 0:
                continue
            centroids[idx] = X[labels == idx].mean(dim=0)
        if torch.allclose(centroids, centroids_old):
            print(f"K-means hội tụ sau {i+1} lần lặp.")
            break

    return centroids, labels