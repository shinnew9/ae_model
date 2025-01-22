import torch

def cov_v_diff(in_v):
    in_v_tmp = in_v.clone()
    mu = torch.mean(in_v_tmp.t(), 1)
    diff = torch.sub(in_v, mu)

    return diff, mu


def cov_v(diff, num):
    var = torch.matmul(diff.t(), diff) / num
    return var


def mahalanobis(u, v, cov_x, use_precision=False, reduction=True):
    num, dim = v.size()
    if use_precision == True:
        inv_cov = cov_x
    else:
        inv_cov = torch.inverse(cov_x)
    delta = torch.sub(u, v)
    m_loss = torch.matmul(torch.matmul(delta, inv_cov), delta.t())

    if reduction:
        return torch.sum(m_loss)/num
    else:
        return m_loss, num


def loss_function_mahala(recon_x, x, block_size, cov=None, is_source_list=None, is_target_list=None, update_cov=False, use_precision=False, reduction=True):
    ### Modified mahalanobis loss###
    if update_cov == False:
        loss = mahalanobis(recon_x.view(-1, block_size), x.view(-1, block_size), cov, use_precision, reduction=reduction)
        return loss
    else:
        diff = x - recon_x
        cov_diff_source, _ = cov_v_diff(in_v=diff[is_source_list].view(-1, block_size))

        cov_diff_target = None
        is_calc_cov_target = any(is_target_list)
        if is_calc_cov_target:
            cov_diff_target, _ = cov_v_diff(in_v=diff[is_target_list].view(-1, block_size))

        loss = diff**2
        if reduction:
            loss = torch.mean(loss, dim=1)
        
        return loss, cov_diff_source, cov_diff_target


def loss_reduction_mahala(loss):
    return torch.mean(loss)


def calc_inv_cov(model, device="cpu", epsilon=1e-5):
    inv_cov_source = None
    inv_cov_target = None

    # Add epsilon to the diagonal to avoid singular matrix
    cov_x_source = model.cov_source.data
    cov_x_source = cov_x_source.to(device).float()
    cov_x_source += torch.eye(cov_x_source.size(0), device=device) * epsilon  # Regularization
    inv_cov_source = torch.inverse(cov_x_source)
    inv_cov_source = inv_cov_source.to(device).float()

    cov_x_target = model.cov_target.data
    cov_x_target = cov_x_target.to(device).float()
    cov_x_target += torch.eye(cov_x_target.size(0), device=device) * epsilon  # Regularization
    inv_cov_target = torch.inverse(cov_x_target)
    inv_cov_target = inv_cov_target.to(device).float()

    return inv_cov_source, inv_cov_target




# 기존 코드에 scaled_loss 함수 추가 - Mahalanobis Distance를 기반으로 계산된 Loss에 Sigmoid를 적용하여 스케일링
# 하는 이유는 Mahalanobis Distance는 정규분포를 따르는 데이터에 대해 거리를 계산하므로, 이를 0~1 사이의 값으로 변환하기 위함
def scaled_loss(recon_x, x, block_size, cov, use_precision=False, reduction=True):
    """
    Mahalanobis Distance를 기반으로 계산된 Loss에 Sigmoid를 적용하여 스케일링.
    Args:
        recon_x (torch.Tensor): Autoencoder 재구성 출력
        x (torch.Tensor): 원본 입력 데이터
        block_size (int): Mahalanobis 계산 시 블록 크기
        cov (torch.Tensor): 역공분산 행렬
        use_precision (bool): True일 경우 역공분산 행렬을 직접 사용
    Returns:
        torch.Tensor: Sigmoid로 스케일링된 Mahalanobis Loss
    """
    # Mahalanobis Loss 계산
    loss = loss_function_mahala(recon_x, x, block_size, cov, use_precision=use_precision, reduction=True)
    
    # Sigmoid를 적용하여 스케일링
    return torch.sigmoid(loss)