from mrirecon.models.primal_dual import PrimalDualNetwork, convert_ncc1701_state_dict, fourier_pd, fourier_pdunet

MODELS = {
    "fourier-pd": fourier_pd,
    "fourier-pdunet": fourier_pdunet,
}


def build_model(name: str, **kwargs) -> PrimalDualNetwork:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


__all__ = ["MODELS", "PrimalDualNetwork", "build_model", "convert_ncc1701_state_dict", "fourier_pd", "fourier_pdunet"]
