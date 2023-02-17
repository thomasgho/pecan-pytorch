import torch


def rotate_point_cloud(data):
    """ 
    Randomly rotate point cloud for augumentation

    Parameters
    ----------
    data : dict
        Dictionary packing pdb id and corresponding graph
    
    Returns
    dict
        Dictionary packing pdb id and corresponding rotated graph
    """
    
    coords = data["graph"].ndata["coord"]
    device = coords.device
    
    rotation_angle_x = torch.rand(1) * 2 * torch.pi
    rotation_angle_y = torch.rand(1) * 2 * torch.pi
    rotation_angle_z = torch.rand(1) * 2 * torch.pi
    
    cos_x = torch.cos(rotation_angle_x)
    cos_y = torch.cos(rotation_angle_y)
    cos_z = torch.cos(rotation_angle_z)
    sin_x = torch.sin(rotation_angle_x)
    sin_y = torch.sin(rotation_angle_y)
    sin_z = torch.sin(rotation_angle_z)
    
    rotation_matrix_x = torch.tensor(
        [
            [1., 0., 0],
            [0., cos_y, -sin_y],
            [0., sin_y, cos_y]
        ]
    ).to(device)
    rotation_matrix_y = torch.tensor(
        [
            [cos_x, 0., sin_x],
            [0., 1., 0.],
            [-sin_x, 0., cos_x]
        ]
    ).to(device)
    rotation_matrix_z = torch.tensor(
        [
            [cos_z, -sin_z, 0.],
            [sin_z, cos_z, 0.],
            [0., 0., 1.]
        ]
    ).to(device)
    
    rotated_coords = (coords 
                      @ rotation_matrix_x.t() 
                      @ rotation_matrix_y.t() 
                      @ rotation_matrix_z.t())
    
    data["graph"].ndata["coord"] = rotated_coords
    
    return data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ 
    Randomly jitter point cloud points. Jittering is per point.

    Parameters
    ----------
    data : dict
        Dictionary packing pdb id and corresponding graph
    
    Returns
    dict
        Dictionary packing pdb id and corresponding jittered graph
    """
    
    assert(clip > 0)
    
    coords = data["graph"].ndata["coord"]
    device = coords.device
    
    N, D = coords.shape
    jitter = torch.clip(sigma * torch.rand(N, D), -1 * clip, clip).to(device)
    
    jittered_coords = coords + jitter
    data["graph"].ndata["coord"] = jittered_coords
    
    return data