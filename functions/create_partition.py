import torch


def create_partition_func_grid(input_size):
    """
    Create a partition function for a tensor with given size.
    The partition function will assign the paritition ID to the input.
    8 dimensions are partitioned, each into 3 equal parts.
    """

    index_list = set()
    while len(index_list) < 8:
        index_list.add(
            tuple(
                torch.randint(0, input_size[i], (1,)).item()
                for i in range(len(input_size))
            )
        )
    index_list = list(index_list)
    index_list = [[slice(None)] + list(x) for x in index_list]

    def _cal_partition(x: torch.Tensor) -> torch.Tensor:
        """
        x: batch of images / ...
        return: partition ID for each image (or sth)
        """

        res = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)
        for index in index_list:
            res = res * 3 + (x[index] > 1 / 3) + (x[index] > 2 / 3)
        return res.long()

    return _cal_partition


def create_partition_func_1nn(input_size, n_centroids=5000):
    """
    Create a partition function for a tensor with given size, which will assign
    the paritition ID to the input.
    """

    def _create_centroids(input_size, n_centroids) -> torch.Tensor:
        """
        Randomly create centroids for partitioning.
        """
        return torch.rand([n_centroids] + list(input_size))

    centroids = _create_centroids(input_size, n_centroids)
    centroids = centroids.flatten(start_dim=1)

    def _cal_partition(x: torch.Tensor) -> torch.Tensor:
        """
        x: batch of images / ...
        return: partition ID for each image (or sth)
        """
        nonlocal centroids
        if x.device != centroids.device:
            centroids = centroids.to(x.device)

        x_flatten = x.flatten(start_dim=1)
        distance_matrix = torch.cdist(x_flatten, centroids)
        res = distance_matrix.argmin(dim=1)
        return res.long()

    return _cal_partition


def create_nearest_buffer_instance_func(buffer: torch.Tensor):
    def nearest_buffer_instance(augment: torch.Tensor):
        ans = torch.cdist(buffer.flatten(start_dim=1), augment.flatten(start_dim=1)).argmin(
            dim=0
        )
        return ans
    return nearest_buffer_instance


def nearest_buffer_instance(buffer: torch.Tensor, augment: torch.Tensor):
    ans = torch.cdist(buffer.flatten(start_dim=1), augment.flatten(start_dim=1))
    print(ans)
    print('pairwise distance shape: ', ans.shape)
    ans = ans.argmax(dim=0)
    return ans



def create_id_func():
    def id(x):
        return torch.Tensor(range(x.shape[0])).long()
    return id


if __name__ == "__main__":
    buffer = torch.rand(100, 3, 32, 32).cuda()
    augment = torch.rand(1000, 3, 32, 32).cuda()
    print(nearest_buffer_instance(buffer, augment))