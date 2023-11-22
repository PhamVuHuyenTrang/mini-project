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


def create_partition_func_1nn(input_size, n_centroids = 5000):
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

def dataid_in_cluster(clusterID):
    '''
    Extracts data indices sorted by clusterID.

    Parameters:
        clusterID (torch.tensor): Tensor containing cluster IDs.

    Returns:
        unique_cluster (torch.tensor): Unique cluster IDs appearing in the data.
        sorted_data (torch.tensor): Data indices in each cluster, sorted in ascending order by clusterID.
        id (torch.tensor): Indices indicating the start and end positions of each cluster.

    Usage with buffer_clusterID or augment_clusterID:
        unique_cluster, sorted_data, id = dataid_in_cluster(clusterID)
        for i in range(len(unique_cluster)):
            data[sorted_data[id[i]:id[i+1]]]
    '''
    # Obtain unique cluster IDs
    unique_cluster = clusterID.unique().reshape(-1, 1)

    # Create a tensor with repeated cluster IDs for comparison
    cluster_repeat = torch.stack([clusterID] * len(unique_cluster), dim=0)

    # Find non-zero indices where the cluster ID matches the unique cluster ID
    nz = (cluster_repeat == unique_cluster).nonzero(as_tuple=True)

    # Identify positions where the cluster transitions occur
    transition_positions = nz[0][[0] + list(range(0, nz[0].shape[0]))] != nz[0][[-1] + list(range(1, nz[0].shape[0])) + [0]]

    return unique_cluster, nz[1], transition_positions.nonzero()
