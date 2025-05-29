import numpy as np


def radius_numpy(x, y, r, ptr_x=None, ptr_y=None, max_num_neighbors=32, ignore_same_index=False):
    """
    用numpy实现radius_cuda算子的功能
    :param x: 节点特征矩阵, shape为 [N, F]
    :param y: 节点特征矩阵, shape为 [M, F]
    :param r: 半径
    :param ptr_x: 可选的批次指针
    :param ptr_y: 可选的批次指针
    :param max_num_neighbors: 每个元素返回的最大邻居数
    :param ignore_same_index: 是否忽略相同索引的点
    :return: 邻接索引
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if ptr_x is None and ptr_y is None:
        # 单示例情况
        n = x.shape[0]
        m = y.shape[0]
        out_vec = []
        for i in range(m):
            distances = np.linalg.norm(x - y[i], axis=1)
            neighbors = []
            for _, t in enumerate(distances):
                if distances[_] <= r:
                    neighbors.append(_)
            neighbors = np.array(neighbors)
            if ignore_same_index:
                neighbors = neighbors[neighbors != i]
            count = 0
            for neighbor in neighbors:
                if count < max_num_neighbors:
                    out_vec.extend([neighbor, i])
                    
                    count += 1
        out = np.array(out_vec).reshape(-1, 2).T
        return out
    else:
        # 批次情况
        out_vec = []
        for b in range(len(ptr_x) - 1):
            x_start, x_end = ptr_x[b], ptr_x[b + 1]
            y_start, y_end = ptr_y[b], ptr_y[b + 1]
            if x_start == x_end or y_start == y_end:
                continue
            for i in range(y_start, y_end):
                distances = np.linalg.norm(x[x_start:x_end] - y[i], axis=1)
                neighbors = []
                for _, t in enumerate(distances):
                    if distances[_] <= r:
                        neighbors.append(_ + x_start)
                neighbors = np.array(neighbors)
                if ignore_same_index:
                    neighbors = neighbors[neighbors != i]
                count = 0
                for neighbor in neighbors:
                    if count < max_num_neighbors:
                        out_vec.extend([neighbor, i])
                        count += 1
        out = np.array(out_vec).reshape(-1, 2).T
        return out


def generate_case(case_id, compute_dtype=np.float32):
    x = np.random.uniform(-5, 5, [100, 2]).astype(compute_dtype)
    y = np.random.uniform(-5, 5, [50, 2]).astype(compute_dtype)
    ptr_x = None
    ptr_y = None
    r = 1.0
    max_num_neighbors = 10
    ignore_same_index = False
    assign_index = radius_numpy(x, y, r, ptr_x, ptr_y, max_num_neighbors, ignore_same_index)
    if ptr_x is None:
        return {"input_desc": {"x": {"shape": list(x.shape), "value": x.tolist()}, 
                        "y": {"shape": list(y.shape), "value": y.tolist()},
                        },
            "attr": {"r": {"value": r}, "max_num_neighbors": {"value": max_num_neighbors},
                    "ignore_same_index": {"value": ignore_same_index}},
            "output_desc": {"out": {"shape": list(assign_index.shape)}}
            }
    else:
        return {"input_desc": {"x": {"shape": list(x.shape), "value": x.tolist()}, 
                        "y": {"shape": list(y.shape), "value": y.tolist()},
                        "ptr_x": {"shape": list(ptr_x.shape), "value": ptr_x.tolist()},
                        "ptr_y": {"shape": list(ptr_y.shape), "value": ptr_y.tolist()},
                        },
                            
            "attr": {"r": {"value": r}, "max_num_neighbors": {"value": max_num_neighbors},
                    "ignore_same_index": {"value": ignore_same_index}},
            "output_desc": {"out": {"shape": list(assign_index.shape)}}
            }


def fuzz_branch_001():
    return generate_case(1)